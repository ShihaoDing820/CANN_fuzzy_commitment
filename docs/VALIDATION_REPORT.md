# 功能验证报告

## 1. 验证信息

```text
日期：2026-11-16
主机：Ubuntu x86_64 虚拟机
CANN：9.0.0
Python：3.11.18
CMake：3.28.3
CPU Debug架构：dav-2201
OPP离线目标：ascend310p
真实NPU：无，npu-smi不可用
```

## 2. 标准 OPP 编译与打包

命令：

```bash
source /home/dsh/cann-dev/activate.sh
cd custom_opp
./build.sh
```

结果：通过。

已实际生成：

- `FuzzyCommitmentAdaMTransCustom` Ascend310P `.o/.json`。
- `FuzzyCommitmentLdpcRecoverCustom` Ascend310P `.o/.json`。
- `libcust_opapi.so`。
- `libcust_opmaster_rt2.0.so`。
- `libcust_opsproto_rt2.0.so`。
- ACLNN 头文件。
- 信息库 JSON。
- `op_proto.h`。
- `custom_opp_ubuntu_x86_64.run`。

最终 `.run` 构建 SHA-256：

```text
d108356ee705511a38be0844c113be957ab8ec37b3e24395b493cc3fb18601e4
```

包完整性检查：通过，SHA-256 checksums are OK。

## 3. 标准 custom kernel CPU Debug

命令：

```bash
./tests/run-standard-kernel-cpu-test.sh
```

结果：

```text
[PASS] standard AdaMTrans kernel: 5 rows on 3 cores, tail covered
[PASS] standard LDPC kernel: 5 blocks on 3 cores, tail covered
[PASS] standard custom-op kernel CPU test completed
```

该测试直接包含并启动 `custom_opp/op_kernel` 的同一份生产源码，不是另写一个
简化 kernel。

覆盖项：

| 用例 | 结果 |
|---|---|
| Ada batch=5/core=3 | 通过 |
| Ada 尾核分配 2/2/1 | 通过 |
| 阈值边界与 mask | 与 Host 逐 bit 一致 |
| LDPC blocks=5/core=3 | 通过 |
| LDPC 尾核分配 2/2/1 | 通过 |
| probe XOR helper | 通过 |
| 合法全零码字 | 正确恢复 |
| decode_ok | 全部为 1 |
| syndrome_weight | 全部为 0 |
| iterations | 全部为 1 |

## 4. LFW 10 对真实特征回归

命令：

```bash
./prototype/run-tests.sh
```

数据选择：

- 前 5 对正样本：LFW pair 0–4。
- 前 5 对负样本：LFW pair 300–304。

结果：

```text
CANN/Host pipeline agreement: 10/10
Biometric classification: 10/10
```

正样本最大 LDPC 迭代为 3–6，均接受；负样本多数达到 50 轮且最终拒绝。
分类结果不是程序通过的唯一条件，CANN 与 Host 参考不一致才会使功能回归失败。

## 5. LFW 6000 对全量回归

命令：

```bash
./prototype/run-all.sh
```

规模：

```text
12000个ArcFace模板
6000对样本
18000个LDPC码块
20个CPU Debug worker
```

结果：

```text
AdaMTrans CANN/Host agreement: 12000/12000
TP=2989 TN=2837 FP=163 FN=11
accuracy=0.971000
GMR=0.996333
FMR=0.054333
FNMR=0.003667
valid_pipeline_pairs=6000/6000
average_ldpc_iterations=22.584722
```

解释：

- `6000/6000` 表示全部流程产生合法输出，不代表全部分类正确。
- 11 个正样本被拒绝。
- 163 个负样本被错误接受。
- FMR 5.4333% 不满足高安全认证要求。

## 6. 已验证与未验证边界

### 已验证

- JSON 可被 CANN 9.0.0 `msOpGen` 解析。
- Host shape/dtype/Tiling 可以编译、链接和打包。
- 两个标准 kernel 可以 CPU Debug 实际执行。
- 多核尾行/尾块分配正确。
- Ascend310P 离线 kernel 可以生成。
- ACLNN、原型、Tiling 库和 `.run` 可以生成。
- LFW 10 对与 6000 对端到端原型可以执行。

### 未验证

- 真实 Ascend310P 运行。
- 其他 SoC。
- 真机 ACLNN 端到端调用。
- 真机性能、带宽和利用率。
- float/迭代结果在所有 CANN 版本间完全一致。
- 生产安全性、模板熵和 helper-data 泄漏。
- 严格训练/测试折隔离后的 FMR/FNMR。

## 7. 建议补充的目标设备测试

1. 在目标设备执行 `npu-smi info`。
2. 按实际 SoC 重新编译 `.run`。
3. 安装自定义 OPP。
4. 使用生成的 ACLNN API 运行 Ada 和 LDPC 单算子。
5. 将输出与本仓库 Host 真值逐 bit 比较。
6. 覆盖 `B/S=1`、整除和尾核、空输入、错误 shape/dtype。
7. 使用 msOpST 或同等工具记录功能与性能。
