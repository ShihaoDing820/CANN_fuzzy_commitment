# Fuzzy Commitment 算法、常见做法与 CANN 开发差异

## 1. 先区分“协议”和“算子”

Fuzzy Commitment 是注册与验证协议，不是一个单独的逐元素函数。完整流程是：

```text
注册：
ArcFace → 二值模板w → 随机LDPC码字c
                     → P=w XOR c
                     → tag=SHA256(c)

验证：
ArcFace' → 二值模板w' → y=w' XOR P
                       → LDPC恢复c_hat
                       → SHA256(c_hat)与tag比较
```

本工程只把适合张量计算的二值化和 LDPC 解码放在 AI Core。数据读取、随机数、
helper data、哈希和协议状态保留在 Host。

## 2. 为什么先把人脸特征二值化

ArcFace 输出 512 个 float，LDPC 处理固定长度 bit 向量。必须定义一个稳定、
可复现的转换：

```text
x<t0          → 000
t0≤x<t1       → 001
t1≤x<t2       → 011
x≥t2          → 111
```

这是温度计式编码。相邻区间只变化一位，例如 `001→011`，比普通二进制编码
更适合含噪声的生物特征。

512 维各产生 3 bit，所以模板长度为 1536 bit。mask 再去掉部分位置：

```text
output = LSSC(embedding) AND mask
```

### 常见做法

- 在独立训练折学习阈值。
- 按每一位的稳定性和区分性选择 mask。
- 将参数固定后再评估测试折。
- 可能用可学习转换或 Neural-MS 联合训练。

### 本工程做法

- 使用全体输入的均值和平均标准差计算三个共享阈值。
- 使用固定种子的可复现 mask。
- 当前实际保留 `840/1536` bit。
- 未训练完整 AdaMTrans/Neural-MS。

因此当前参数适合工程回归，不是严格生物识别评估参数。正式实验必须按
train/test fold 隔离阈值和 mask。

## 3. Fuzzy Commitment 为什么能容忍误差

注册模板为 `w`，合法纠错码字为 `c`：

```text
P=w XOR c
```

验证模板为 `w'`：

```text
w' XOR P = w' XOR w XOR c = error XOR c
```

两次采集越接近，`error` 的 Hamming weight 越低。LDPC 的任务就是从
`c XOR error` 恢复 `c`。

helper `P` 通常可以公开，但“可以公开”不代表“没有信息泄漏”。纠错码参数、
模板熵、不同应用之间的可链接性和 helper 复用都需要正式分析。

## 4. 为什么拆成三个 520-bit 块

二值模板长度为 1536：

```text
1536 = 3 × 512
```

当前 BG2/Z=10 LDPC 的码长为 520，所以每个 512-bit 块补 8 个零。补零只用于
尺寸适配，不增加熵。注册和验证必须使用完全相同的补零规则，否则会人为产生
差异。

## 5. LDPC 中 VN、CN、边和 K 是什么

```text
N/VN=520：码字有520个变量bit
M/CN=420：有420条奇偶校验方程
E=1970：变量与校验方程之间有1970条连接
K=100：H秩为420时，零空间维数为520-420=100
```

BG2 基图是 42×52，提升因子 Z=10 后变成 420×520 的稀疏图。

代码保存两个方向的 CSR 风格索引：

```text
checkOffsets/checkVariables：从CN查VN
variableOffsets/variableEdges：从VN查消息边
```

因为一次迭代既要执行 CN 更新，也要执行 VN 更新。保存两个方向可以避免每轮
重建反向连接。

## 6. Normalized Min-Sum 做什么

收到 bit 后先建立置信度：

```text
0 → +4.0
1 → -4.0
```

每轮执行：

1. CN 更新：计算符号积、最小绝对值和第二小绝对值。
2. 乘归一化系数 0.75，形成 CN→VN 消息。
3. VN 更新：累加 channel LLR 与所有关联 CN 消息。
4. 根据总 LLR 正负进行硬判决。
5. 检查 syndrome，等于 0 时提前结束。

最小值/第二小值用于排除当前边自身的输入。Normalized Min-Sum 比精确
Sum-Product 简单，硬件代价低，但参数需要针对数据和码结构调优。

常见解码器还包括：

- Sum-Product。
- 原始 Min-Sum。
- Offset Min-Sum。
- Layered Min-Sum。
- Neural Min-Sum。

本工程是固定系数 Normalized Min-Sum。

## 7. 为什么 SHA-256 放在 Host

SHA-256、CSPRNG 和模板状态管理不是 AI Core 的优势。AI Core 更适合规则的
批量张量计算，把这些逻辑强行放入 kernel 会增加实现和安全风险。

当前 Host 保存注册码字的 SHA-256，验证时对恢复码字重新哈希并用
`CRYPTO_memcmp` 做常量时间比较。

注意：哈希不能让低熵生物模板自动变成高熵秘密。它只用于确认是否恢复了同一
码字。

## 8. 接口为什么使用 ND、float32、uint16

ArcFace 特征是普通二维矩阵 `[B,512]`，不是带通道和空间维的图像，因此使用
`ND`，而不是 NCHW/NHWC。

当前类型：

```text
ArcFace/threshold/LLR：float32
bit模板/码字：uint16
图索引/元数据：int32
```

每个 bit 使用 `uint16` 会浪费内存，但实现和调试直接。常见优化是 `uint8`
或 bit packing；代价是解包、对齐、向量化和 API 复杂度增加。首版优先保证
结果清晰可验证。

## 9. 为什么要有 decode_ok、syndrome_weight、iterations

只输出 `decoded` 很难判断失败原因。额外输出可以回答：

- 是否满足全部校验方程。
- 还剩多少条校验不满足。
- 是否达到最大迭代。
- 某批数据为什么变慢。

这些字段对功能测试和真机 profiling 很重要。协议层不应只看 decoded 内容，
还应要求三个块都成功解码。

## 10. 普通算子与 CANN custom op 的差异

| 项目 | 普通 CPU/C++ | CUDA/PyTorch 扩展 | CANN custom op |
|---|---|---|---|
| 设备 | CPU | NVIDIA GPU | 昇腾 AI Core |
| 并行单位 | 线程 | thread/block/grid | AI Core/blockDim/tile |
| 内存 | 缓存大多隐式 | global/shared/register | GM/UB/LocalTensor 显式管理 |
| 接口 | 函数签名 | framework schema | JSON、OpDef、信息库 |
| 动态切分 | 循环/线程池 | launch 参数 | Host Tiling/TilingData |
| 发布 | executable/so | wheel/so | custom OPP `.run/.so` |
| 兼容性 | CPU ABI | CUDA/SM | CANN 版本和 SoC |
| 入图 | 通常无 | PyTorch 注册 | 还需 shape/dtype 和框架适配 |

一个完整 CANN 算子不是只有 kernel，而是：

```text
原型JSON
+ Host注册
+ shape/dtype推导
+ Tiling
+ AI Core kernel
+ 信息库
+ ACLNN API
+ 打包部署
+ 测试与性能报告
```

## 11. Host Tiling 为什么必要

AI Core 不适合在运行时做大量资源规划。Host 先计算：

- batch 有多少。
- 设备有多少核心。
- 每个核心处理几行或几个码块。
- 是否存在尾核。
- 属性是否合法。
- workspace 多大。

Kernel 只读取小型 TilingData 并执行。

AdaMTrans 可按行独立切分。LDPC 单个码块内部有迭代依赖，所以本工程只在码块
之间并行，不把同一个块的边随意分散到多个核心。

## 12. GM、UB 与 CopyIn/Compute/CopyOut

CANN 中 GM 是全局设备内存，UB 是核心上的高速局部内存。高性能 kernel 通常：

```text
CopyIn：GM→UB
Compute：在UB计算
CopyOut：UB→GM
```

AdaMTrans 已按这一结构组织。LDPC 也把消息和硬判决放入 UB，但稀疏图索引仍
从 GM 访问，且 520×uint16 不满足整 32-byte 长度，因此当前逐元素安全搬入。
这保证了边界正确，但不代表已经达到真机最优性能。

## 13. 当前实现与常见生产实现对比

| 项目 | 常见生产做法 | 当前实现 |
|---|---|---|
| 阈值 | 训练折学习 | 全量输入统计 |
| mask | 稳定性学习/可撤销 | 固定可复现 mask |
| RNG | CSPRNG | 固定种子 `mt19937` 测试 |
| LDPC | 数据调参/训练 | BG2/Z=10 固定 NMS |
| bit 存储 | uint8/bit pack | uint16 |
| API | 安装后 ACLNN/入图 | 已生成 ACLNN，原型也可直启 kernel |
| Tiling | 按目标 SoC 动态 | 已实现按核心动态分配 |
| NPU 性能 | 真机 profiling | 尚无 NPU |
| 安全评估 | entropy/leakage/linkability | 尚未完成 |

## 14. 测试结果应怎样理解

标准 kernel CPU Debug 测试验证的是：

- 新 custom op kernel 确实能执行。
- 多核尾行和尾块没有漏算。
- Ada 输出与 Host 真值逐 bit 相同。
- LDPC 能恢复合法零码字并输出正确元数据。

10 对 LFW 测试还逐 bit 比较 CANN 与 Host 参考解码。

6000 对结果：

```text
TP=2989 TN=2837 FP=163 FN=11
GMR=0.996333 FMR=0.054333 FNMR=0.003667
```

`pipeline 6000/6000` 表示 6000 对都完成了合法执行，不是分类全部正确。FMR
5.4333% 对认证系统过高，不能作为生产参数。

## 15. 最重要的注意事项

1. 先用 `npu-smi info` 确认目标 SoC，再生成二进制。
2. CPU Debug 只能验证功能，不能用于报告 NPU 性能。
3. CANN/CANNLab 版本变化后要重新验证 schema、API、安装路径和二进制。
4. shape、dtype、format 和属性必须在 Host 拒绝非法值，不能让 kernel 越界。
5. DataCopy 必须考虑字节对齐和真实内存边界。
6. TilingData Host 与 Kernel 的字段顺序、类型和大小必须完全一致。
7. 二进制发布必须与 SoC 匹配；源码发布仍需在目标环境重新编译。
8. 不要把固定 mask 当密钥。
9. 不要把 `mt19937` 用于生产注册。
10. helper data、tag、模板和日志都需要访问控制和生命周期管理。
