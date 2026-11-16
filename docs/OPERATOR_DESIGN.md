# Fuzzy Commitment：CANN 自定义算子设计与实现

## 1. 目标与实现范围

本工程把完整 Fuzzy Commitment 协议拆为两个 AI Core 自定义算子和一个 Host
协议层：

1. `FuzzyCommitmentAdaMTransCustom`
   - ArcFace 512 维浮点特征转为 1536-bit 二值模板。
2. `FuzzyCommitmentLdpcRecoverCustom`
   - 使用 BG2/Z=10 Normalized Min-Sum 恢复 520-bit LDPC 码字。
3. Host 协议层
   - 数据读取、阈值和 mask 管理、LDPC 图构造、随机码字、helper data、
     SHA-256 和最终接受/拒绝。

选择这个边界，是因为 AI Core 适合批量张量计算；文件处理、CSPRNG、密码学、
密钥生命周期和协议控制更适合 Host。完整协议不是一个必须全部塞入单个 kernel
的操作。

## 2. 数学定义

### 2.1 LSSC/AdaMTrans 思路

输入：

```text
X ∈ R^(B×512)
t0 < t1 < t2
```

对每个特征值：

```text
level = [x≥t0] + [x≥t1] + [x≥t2]
q(x) = ([level≥3], [level≥2], [level≥1])
```

四个区间依次编码为：

```text
000, 001, 011, 111
```

最后应用 mask：

```text
W[b,j] = q(X[b])[j] AND mask[j]
W ∈ {0,1}^(B×1536)
```

### 2.2 Fuzzy Commitment

每个 1536-bit 模板拆为三个 512-bit 块，各补 8 个零：

```text
w_i ∈ {0,1}^520, i=0,1,2
```

注册：

```text
c_i ← 随机合法LDPC码字，满足 Hc_i^T=0
P_i = w_i XOR c_i
tag = SHA256(c_0||c_1||c_2)
```

验证：

```text
y_i = w'_i XOR P_i
    = c_i XOR (w'_i XOR w_i)
c_hat_i = LDPCDecode(y_i)
accept ⇔ 所有码块解码有效且
         SHA256(c_hat_0||c_hat_1||c_hat_2)=tag
```

### 2.3 LDPC 参数

| 参数 | 数值 |
|---|---:|
| 基图 | 5G-NR BG2 Set 2 |
| 基图尺寸 | 42×52 |
| 提升因子 Z | 10 |
| 变量节点 N/VN | 520 |
| 校验节点 M/CN | 420 |
| 自由度 K | 100 |
| Tanner 图边数 E | 1970 |
| 最大迭代 | 默认 50 |
| 初始 LLR | 默认 ±4.0 |
| Min-Sum 归一化 | 默认 0.75 |

本实现复用 WiFaKey 的 BG2/Z=10 结构，但使用固定参数 Normalized Min-Sum，
不声称复现完整 Neural-MS 训练过程。

## 3. 算子原型 JSON

原型位于：

```text
specs/fuzzy_commitment_ops.json
```

JSON 已由 CANN 9.0.0 `msOpGen` 实际解析。每个输入输出的 `type` 与 `format`
数组长度一致并按位置对应。本工程只声明实际实现的 dtype/format，不虚报
float16、NCHW 或广播支持。

生成命令：

```bash
msopgen gen \
  -i specs/fuzzy_commitment_ops.json \
  -op FuzzyCommitmentAdaMTransCustom \
  -c ai_core-ascend310p \
  -lan cpp \
  -out custom_opp

msopgen gen \
  -i specs/fuzzy_commitment_ops.json \
  -op FuzzyCommitmentLdpcRecoverCustom \
  -c ai_core-ascend310p \
  -lan cpp \
  -m 1 \
  -out custom_opp
```

第二条使用 `-m 1` 将第二个算子加入同一工程。

## 4. AdaMTrans 算子接口

### 4.1 输入输出

| 名称 | 方向 | dtype | format | shape |
|---|---|---|---|---|
| `embeddings` | input | float32 | ND | `[B,512]` |
| `thresholds` | input | float32 | ND | `[3]` |
| `mask` | input | uint16 | ND | `[1536]` |
| `binary_templates` | output | uint16 | ND | `[B,1536]` |

属性：

```text
embedding_dim=512
bits_per_feature=3
```

属性保留在接口中用于明确算法契约。当前 kernel 是固定结构实现，Host 会拒绝
非 512/3 的属性，而不是让接口属性与实际算法不一致。

### 4.2 Shape 推导

Host 检查：

```text
rank(embeddings)=2
embeddings.shape[1]=512
thresholds.shape=[3]
mask.shape=[1536]
B可以在图构建阶段未知，但运行时必须大于0
```

输出：

```text
[B,1536]
```

输出 dtype 固定为 `uint16`，不会错误继承 float32 输入 dtype。

### 4.3 Tiling

TilingData：

```cpp
struct FuzzyCommitmentAdaMTransCustomTilingData {
    uint32_t batchSize;
    uint32_t baseRowsPerCore;
    uint32_t extraRowCoreCount;
    uint32_t usedCoreNum;
    uint32_t tileRows;
    uint32_t embeddingDim;
    uint32_t bitsPerFeature;
};
```

Host 从平台信息读取可用核心数：

```text
usedCoreNum = min(B, availableCoreNum)
baseRowsPerCore = B / usedCoreNum
extraRowCoreCount = B % usedCoreNum
```

前 `extraRowCoreCount` 个核各多处理一行。因此 `B=5, core=3` 时分配为
`2/2/1`，不存在漏算尾行。

### 4.4 Kernel

入口参数顺序为：

```text
embeddings, thresholds, mask, binaryTemplates, workspace, tiling
```

主要结构：

```text
Init
  → 绑定GM
  → 计算本核起始行与行数
  → 初始化输入队列、输出队列和mask UB
Process
  → CopyIn：一行embedding从GM搬到UB
  → Compute：阈值比较、LSSC编码、mask
  → CopyOut：1536bit结果写回GM
```

embedding 每行 2048 bytes、mask/output 每行 3072 bytes，满足当前简单
`DataCopy` 对齐要求。当前 Compute 使用 UB 上的标量访问保证正确性；真机性能版
还可继续使用 Compare/Select/逻辑向量指令优化。

## 5. LDPC Recover 算子接口

### 5.1 输入输出

| 名称 | 方向 | dtype | format | shape |
|---|---|---|---|---|
| `probe` | input | uint16 | ND | `[S,520]` |
| `helper` | input | uint16 | ND | `[S,520]` |
| `check_offsets` | input | int32 | ND | `[421]` |
| `check_variables` | input | int32 | ND | `[1970]` |
| `variable_offsets` | input | int32 | ND | `[521]` |
| `variable_edges` | input | int32 | ND | `[1970]` |
| `decoded` | output | uint16 | ND | `[S,520]` |
| `decode_ok` | output | int32 | ND | `[S]` |
| `syndrome_weight` | output | int32 | ND | `[S]` |
| `iterations` | output | int32 | ND | `[S]` |

属性：

```text
max_iterations=50
channel_llr=4.0
normalization=0.75
```

Host 限制：

```text
1 ≤ max_iterations ≤ 1000
channel_llr > 0
0 < normalization ≤ 1
```

### 5.2 图结构

四个稀疏数组提供 Tanner 图的两个方向：

```text
check_offsets/check_variables：CN → VN
variable_offsets/variable_edges：VN → Edge
```

CN 更新和 VN 更新分别需要不同遍历方向。两个方向都保存可以避免每轮扫描
整个图重新建立反向关系。

### 5.3 Tiling

```cpp
struct FuzzyCommitmentLdpcRecoverCustomTilingData {
    uint32_t blockCount;
    uint32_t baseBlocksPerCore;
    uint32_t extraBlockCoreCount;
    uint32_t usedCoreNum;
    uint32_t maxIterations;
    float channelLlr;
    float normalization;
};
```

LDPC 每轮存在全图 CN/VN 消息依赖。因此首版不把同一个码块拆到多个核心，
而是：

```text
一个完整码块由一个核心处理
多个独立码块在核心之间并行
```

这优先保证正确性和确定性，也避免引入跨核同步。后续性能优化可研究 layered
调度，但必须重新验证结果和数值顺序。

### 5.4 LocalTensor

每个核心分配：

```text
probe/helper          2 × 520 × uint16
channel_llr            520 × float32
q_message             1970 × float32
r_message             1970 × float32
hard_decision          520 × uint16
```

`probe/helper` 的单块大小是 1040 bytes，不是完整的 32-byte DMA 倍数。当前
正确性实现逐元素搬入，避免简单 DataCopy 越界读取尾部 8 bytes。真机优化可以
使用 `DataCopyPad` 或显式 padding，但必须保证实际分配边界安全。

### 5.5 解码流程

```text
received = probe XOR helper
LLR = received==0 ? +channel_llr : -channel_llr
初始化q/r
循环最多max_iterations：
  CN更新：符号积、最小值、第二小值、normalization
  VN更新：channel LLR + 关联r消息
  硬判决
  syndrome检查
  syndrome=0时提前结束
输出decoded/decode_ok/syndrome_weight/iterations
```

额外元数据用于测试、监控和失败分析。生产协议层应要求所有三个码块
`decode_ok=1` 后再执行最终 tag 接受逻辑。

## 6. Host 与 Kernel 的职责

### Host

- 输入 shape/dtype/format 校验。
- 动态输出 shape 推导。
- 属性范围校验。
- 获取平台核心数。
- 计算行/码块分配。
- workspace 设置。
- 随机数、helper data、SHA-256 和最终协议逻辑。

### Kernel

- 只消费已经确定的张量和 TilingData。
- 不解析文件。
- 不动态分配 C++ 容器。
- 不生成安全随机数。
- 不负责保存用户模板或协议状态。

当前两个 kernel 均不需要跨核临时数据，因此外部 workspace 为 0。UB
`TBuf/TQue` 不属于 ACLNN 外部 workspace。

## 7. 编译与交付产物

`custom_opp/CMakePresets.json` 当前配置：

```text
ASCEND_COMPUTE_UNIT=ascend310p
ASCEND_CANN_PACKAGE_PATH=/home/dsh/Ascend/cann-9.0.0
ENABLE_SOURCE_PACKAGE=True
ENABLE_BINARY_PACKAGE=True
vendor_name=customize
```

`./build.sh` 已生成：

- 两个 Ascend310P kernel `.o/.json`。
- `libcust_opapi.so`。
- `libcust_opmaster_rt2.0.so` Tiling 库。
- `libcust_opsproto_rt2.0.so` 原型库。
- 算子信息库 JSON。
- `op_proto.h`。
- 两个 ACLNN 头文件。
- `custom_opp_ubuntu_x86_64.run`。

精选产物位于 `artifacts/ascend310p`，完整可再生构建目录不提交 Git。

## 8. 调用边界

推荐协议调用顺序：

```text
Host准备embedding/threshold/mask
→ ACLNN调用AdaMTransCustom
→ Host按三个512bit块补零到520bit
→ 注册：生成码字/helper/tag
→ 验证：准备probe/helper和图数组
→ ACLNN调用LdpcRecoverCustom
→ Host检查decode_ok并计算SHA-256
→ 常量时间比较tag
```

如需入图，只建议把两个纯张量算子入图。CSPRNG、模板存储、helper 管理、
SHA-256 和接受/拒绝留在图外，并在框架适配说明中明确边界。

## 9. 已知限制

1. 当前二进制目标是 Ascend310P，其他 SoC 必须重编译。
2. 当前虚拟机无 NPU，尚无真机功能和性能结果。
3. Ada Compute 尚未完全向量化。
4. LDPC 稀疏图和迭代仍有大量标量访问，性能必须真机测量。
5. 图索引值本身由调用者负责保证在合法范围；生产 Host 应在加载图时验证。
6. `uint16` 表示单 bit 便于开发，但内存效率低于 bit packing。
7. 当前安全参数和 FMR 不满足生产认证要求。
8. 当前测试用 RNG、阈值和 mask 不可直接用于生产注册。

## 10. 完成状态

| 项目 | 状态 |
|---|---|
| 原型 JSON | 完成 |
| msOpGen 工程 | 完成 |
| Host 原型注册 | 完成 |
| shape/dtype 推导 | 完成 |
| TilingData/Tiling | 完成 |
| AdaMTrans kernel | 完成并通过 CPU Debug |
| LDPC kernel | 完成并通过 CPU Debug |
| `.run/.so/op_proto.h/信息库` | 已生成 |
| LFW 10/6000 对回归 | 已通过 |
| 真实 NPU 功能 | 待目标设备验证 |
| 真实 NPU 性能 | 待目标设备验证 |
| 生产安全参数 | 待训练、评估与审计 |
