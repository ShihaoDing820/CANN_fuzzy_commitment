# CANN Fuzzy Commitment

本仓库实现基于 ArcFace、LSSC/AdaMTrans 思路和 5G-NR BG2 LDPC 的
Fuzzy Commitment，并同时提供：

- 可由 `msOpGen` 生成、编译和打包的标准 CANN custom OPP 工程。
- 两个 Ascend C 算子：AdaMTrans 二值化与 LDPC 恢复。
- Host shape/dtype 推导、属性校验和多核 Tiling。
- 标准 kernel 的 CANN CPU Debug 测试。
- 真实 LFW ArcFace 特征的 10 对与 6000 对端到端回归程序。
- Ascend310P 的 `.run/.so/op_proto.h/信息库/kernel` 构建产物。
- 算法、开发、部署、验证、性能边界和安全限制说明。

## 当前状态

| 项目 | 状态 |
|---|---|
| CANN 版本 | 9.0.0 |
| Python | 3.11 |
| 原型 JSON | 已完成并通过 `msOpGen` 校验 |
| Host shape/dtype/Tiling | 已完成并编译通过 |
| AdaMTrans custom kernel | 已完成 |
| LDPC Recover custom kernel | 已完成 |
| Ascend310P 离线 kernel | 已生成 |
| Host/Tiling/ACLNN 库 | 已生成 |
| `.run` 包 | 已生成且完整性校验通过 |
| 标准 kernel CPU 仿真 | 已通过 |
| LFW 10 对回归 | 已通过，CANN/Host `10/10` |
| LFW 6000 对回归 | 已通过，pipeline `6000/6000` |
| 真实 NPU 功能/性能 | 未测试：当前虚拟机没有 `npu-smi` |

“CPU 仿真通过”和“Ascend310P 二进制编译成功”不等于已经在真实 NPU 上
验证性能。部署前必须在目标设备执行 `npu-smi info` 并按实际 SoC 重新构建。

## 仓库结构

```text
CANN_fuzzy_commitment/
├── custom_opp/                 # msOpGen 标准工程、Host/Tiling/Kernel
├── specs/                      # 两个算子的原型 JSON
├── tests/                      # 标准 custom kernel CPU Debug 测试
├── prototype/                  # LFW 10 对和 6000 对端到端原型
├── artifacts/ascend310p/       # 精选构建产物及 SHA-256
├── docs/                       # 设计、算法、部署和验证说明
└── run-all-checks.sh           # 一键构建和回归入口
```

## 一键验证

标准 OPP 编译、标准 kernel CPU 仿真和 10 对 LFW 回归：

```bash
./run-all-checks.sh
```

同时运行全量 6000 对：

```bash
./run-all-checks.sh --full
```

LFW 特征默认从以下位置读取：

```text
/home/dsh/下载/embeddings_arcface_r100_lfw.csv
```

标签默认从以下位置读取：

```text
/home/dsh/cann-dev/biometric-data/lfw_issame.csv
```

数据集体积较大，不包含在 Git 仓库中。可将自定义路径传给原型脚本：

```bash
./prototype/run-tests.sh /path/to/embeddings.csv /path/to/lfw_issame.csv
./prototype/run-all.sh /path/to/embeddings.csv /path/to/lfw_issame.csv
```

## 单独构建标准 OPP

```bash
source /path/to/CANN/set_env.sh
cd custom_opp
./build.sh
```

成功后生成：

```text
custom_opp/build_out/custom_opp_ubuntu_x86_64.run
```

本次 Ascend310P 构建的精选产物也保存在：

```text
artifacts/ascend310p/
```

这些二进制只适用于构建时配置的 SoC。其他昇腾芯片必须修改
`custom_opp/CMakePresets.json`、Host `AddConfig` 和生成目标后重新编译。

## 两个算子

### FuzzyCommitmentAdaMTransCustom

```text
embeddings       float32 [B,512]
thresholds       float32 [3]
mask             uint16  [1536]
binary_templates uint16  [B,1536]
```

完成四区间 LSSC 编码和 mask。Host 按 batch 与可用核心数切分行，kernel 使用
GM→UB→Compute→GM 流程。

### FuzzyCommitmentLdpcRecoverCustom

```text
probe/helper       uint16 [S,520]
LDPC稀疏图          int32  [421]/[1970]/[521]/[1970]
decoded            uint16 [S,520]
decode_ok          int32  [S]
syndrome_weight    int32  [S]
iterations         int32  [S]
```

完成 BG2、Z=10、Normalized Min-Sum 解码。一个完整码块保留在一个核心上，
独立码块之间并行，避免跨核破坏 CN/VN 迭代依赖。

## 文档入口

- [算子设计与实现](docs/OPERATOR_DESIGN.md)
- [算法解释、常见做法与 CANN 差异](docs/ALGORITHM_EXPLANATION.md)
- [构建、部署和调用](docs/BUILD_DEPLOY_USE.md)
- [功能验证报告](docs/VALIDATION_REPORT.md)
- [性能报告与后续真机计划](docs/PERFORMANCE_REPORT.md)

## 安全与评估边界

- 当前阈值由全部 LFW 输入统计，不是严格训练/测试隔离。
- 当前固定 mask 用于复现，不是秘密或可撤销模板机制。
- 原型使用固定种子的 `std::mt19937`，生产注册必须换成 CSPRNG。
- 全量结果 FMR 为 `5.4333%`，不能直接用于生产认证。
- helper data、模板熵、可链接性和侧信道仍需正式安全分析。
- 当前工作复用了 WiFaKey 的 BG2/Z=10 结构和思路，不声称复现完整 Neural-MS。

第三方来源和版本信息见 [THIRD_PARTY_NOTICES.md](prototype/THIRD_PARTY_NOTICES.md)。
