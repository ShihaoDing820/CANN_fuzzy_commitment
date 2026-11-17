# 构建、部署与调用说明

## 1. 环境要求

本仓库验证环境：

```text
CANN 9.0.0
Python 3.11.18
CMake 3.28.3
bisheng: CANN 9.0.0
Host: Ubuntu x86_64
离线二进制目标: Ascend310P
```

目标设备首先执行：

```bash
npu-smi info
```

记录 `Chip Name` 并映射到当前 CANN 支持的 `soc_version`。

## 2. 准备 CANN 环境

```bash
source /path/to/Ascend/cann/set_env.sh
which msopgen
which bisheng
python3 --version
cmake --version
```

`CMakePresets.json` 默认从已激活环境的 `ASCEND_HOME_PATH` 读取 CANN 路径。
如果环境脚本没有设置该变量，应先正确加载 CANN，而不是把其他机器的绝对路径
写死到工程中。

## 3. 从 JSON 重新生成工程

通常不需要重复生成，因为生成后的工程已在仓库中。若要验证生成过程：

```bash
msopgen gen \
  -i specs/fuzzy_commitment_ops.json \
  -op FuzzyCommitmentAdaMTransCustom \
  -c ai_core-ascend310p \
  -lan cpp \
  -out /tmp/FuzzyCommitmentCustom

msopgen gen \
  -i specs/fuzzy_commitment_ops.json \
  -op FuzzyCommitmentLdpcRecoverCustom \
  -c ai_core-ascend310p \
  -lan cpp \
  -m 1 \
  -out /tmp/FuzzyCommitmentCustom
```

`msOpGen` 会拒绝组或其他用户可写的输入 JSON/目录。如果出现安全权限错误：

```bash
chmod 0755 specs
chmod 0644 specs/fuzzy_commitment_ops.json
```

不要在重新生成时直接覆盖已经实现的 `custom_opp/op_host` 和 `op_kernel`。

## 4. 构建标准 custom OPP

```bash
source /path/to/CANN/set_env.sh
cd custom_opp
./build.sh
```

构建会清理并重建 `custom_opp/build_out`，生成：

```text
custom_opp_ubuntu_x86_64.run
libcust_opapi.so
libcust_opmaster_rt2.0.so
libcust_opsproto_rt2.0.so
op_proto.h
ACLNN头文件
算子信息库JSON
Ascend310P kernel .o/.json
```

精选产物位于 `artifacts/ascend310p`。校验：

```bash
cd artifacts/ascend310p
sha256sum -c SHA256SUMS
```

## 5. 目标 SoC 不是 Ascend310P 时

至少需要修改：

1. `custom_opp/CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT`。
2. 两个 `op_host/*.cpp` 的 `AddConfig("ascend310p")`。
3. 重新使用目标 `ai_core-<soc_version>` 验证 `msOpGen`。
4. 重新执行 `./build.sh`。
5. 在目标 NPU 上做功能和性能验证。

不能把 `dav-2201` CPU Debug 架构当成设备 Chip Name，也不能把 Ascend310P
`.o` 复制到不同 SoC 后直接假设可用。

## 6. 安装 `.run` 包

先加载 CANN 环境：

```bash
source /path/to/CANN/set_env.sh
```

查看包信息和校验完整性：

```bash
./artifacts/ascend310p/custom_opp_ubuntu_x86_64.run --info
./artifacts/ascend310p/custom_opp_ubuntu_x86_64.run --check
```

默认安装到当前 CANN OPP 下的 `vendors/customize`：

```bash
./artifacts/ascend310p/custom_opp_ubuntu_x86_64.run
```

如果默认目录不可写，使用包帮助中给出的 `--install-path` 安装到用户目录，并
按安装日志设置：

```bash
export ASCEND_CUSTOM_OPP_PATH=/your/custom/opp/path:${ASCEND_CUSTOM_OPP_PATH:-}
```

默认安装后还应设置 ACLNN 自定义 API 库：

```bash
export LD_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
```

安装路径和环境变量以 `.run` 最终输出为准。

## 7. 标准 kernel CPU Debug 测试

该测试启动的就是 `custom_opp/op_kernel` 中的生产 kernel 源码：

```bash
./tests/run-standard-kernel-cpu-test.sh
```

覆盖：

- AdaMTrans 5 行分配到 3 核，含尾行。
- Ada 输出与 Host 真值逐 bit 比较。
- LDPC 5 块分配到 3 核，含尾块。
- probe/helper XOR 路径。
- 合法全零码字恢复。
- `decode_ok/syndrome_weight/iterations`。

## 8. LFW 端到端原型

10 对：

```bash
./prototype/run-tests.sh \
  /path/to/embeddings_arcface_r100_lfw.csv \
  /path/to/lfw_issame.csv
```

6000 对：

```bash
./prototype/run-all.sh \
  /path/to/embeddings_arcface_r100_lfw.csv \
  /path/to/lfw_issame.csv
```

数据要求：

- embeddings：12000 行，每行 512 个逗号分隔 float。
- 每两行组成一对。
- labels：6000 个 0/1 标签。
- 特征与标签顺序一致。

## 9. ACLNN 调用顺序

构建会生成：

```text
aclnn_fuzzy_commitment_ada_m_trans_custom.h
aclnn_fuzzy_commitment_ldpc_recover_custom.h
```

调用 AdaMTrans 的基本顺序：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;

aclnnFuzzyCommitmentAdaMTransCustomGetWorkspaceSize(
    embeddingsTensor,
    thresholdsTensor,
    maskTensor,
    512,
    3,
    outputTensor,
    &workspaceSize,
    &executor);

// workspaceSize为0时workspace可以为空，否则使用aclrtMalloc分配。
aclnnFuzzyCommitmentAdaMTransCustom(
    workspace,
    workspaceSize,
    executor,
    stream);
```

调用 LDPC：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;

aclnnFuzzyCommitmentLdpcRecoverCustomGetWorkspaceSize(
    probeTensor,
    helperTensor,
    checkOffsetsTensor,
    checkVariablesTensor,
    variableOffsetsTensor,
    variableEdgesTensor,
    50,
    4.0,
    0.75,
    decodedTensor,
    decodeOkTensor,
    syndromeWeightTensor,
    iterationsTensor,
    &workspaceSize,
    &executor);

aclnnFuzzyCommitmentLdpcRecoverCustom(
    workspace,
    workspaceSize,
    executor,
    stream);
```

以上代码省略了 `aclInit`、设备/stream 初始化、`aclTensor` 创建、H2D/D2H、
错误码检查和资源释放。实际程序必须检查每一个 ACL/ACLNN 返回值。

## 10. 完整协议调用顺序

注册：

1. 调用 `AdaMTransCustom(enrollmentEmbedding)`。
2. 将 $3\times512$ bit 拆分并补零为 $3\times520$ bit。
3. 使用 CSPRNG 生成三个合法 LDPC 码字 $\mathbf{c}_0,\mathbf{c}_1,\mathbf{c}_2$。
4. 对每个码块计算
   $\mathbf{P}_i=\mathbf{w}_i\oplus\mathbf{c}_i$，其中 $i\in\{0,1,2\}$。
5. 计算
   $\mathrm{tag}=\operatorname{SHA\!-\!256}(\mathbf{c}_0\mathbin\Vert\mathbf{c}_1\mathbin\Vert\mathbf{c}_2)$。
6. 安全保存 helper、tag 和参数版本。

验证：

1. 调用 `AdaMTransCustom(probeEmbedding)`。
2. 调用 `LdpcRecoverCustom(probe, helper, graph)`。
3. 要求三个码块均满足 $\operatorname{decode\_ok}_i=1$。
4. 计算
   $\widehat{\mathrm{tag}}=\operatorname{SHA\!-\!256}(\hat{\mathbf{c}}_0\mathbin\Vert\hat{\mathbf{c}}_1\mathbin\Vert\hat{\mathbf{c}}_2)$。
5. 常量时间比较 $\widehat{\mathrm{tag}}$ 与注册阶段保存的 $\mathrm{tag}$。

## 11. 入图说明

若接入 MindSpore、ONNX 或其他框架：

- 注册两个张量算子的框架适配。
- 保留已有 shape/dtype 推导规则。
- 将属性传递到 Host Tiling。
- 说明只支持 ND 和当前 dtype。
- 明确 SHA-256、CSPRNG、模板数据库和接受逻辑不入图。
- 对动态 batch 做框架侧测试。

## 12. 卸载

安装包会在 vendor 目录生成卸载脚本。使用实际安装日志给出的卸载路径执行，
不要直接递归删除整个 CANN OPP 目录，以免破坏其他算子。
