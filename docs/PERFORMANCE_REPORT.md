# 性能报告与真机验证计划

## 1. 当前结论

当前环境没有可用 NPU，也没有 `npu-smi`。因此本仓库没有给出虚假的 NPU
时延、吞吐量、AI Core 利用率或带宽数据。

已记录的 CPU Debug 时间只能说明功能测试可执行，不能用于评价 Ascend NPU
性能，也不能与 CPU、CUDA 或其他 NPU 做性能对比。

## 2. 本次 CPU Debug 观察值

全量 6000 对程序内部计时：

```text
elapsed_seconds=8.742205
average_ldpc_iterations=22.584722
```

该时间来自 CPU Debug 模拟环境，其中 20 个 worker 处理 12000 个模板和
18000 个 LDPC 块。模拟器的调度、内存和指令执行方式与真实 AI Core 不同，
所以这不是产品性能数据。

## 3. 当前实现的性能特征

### AdaMTrans

已完成：

- 按 batch 动态分核。
- 尾行无遗漏。
- embedding/mask/output 使用 UB。
- 明确 CopyIn/Compute/CopyOut。

仍可优化：

- 当前阈值比较和编码主要是 UB 标量访问。
- 可评估 Compare/Select/逻辑向量指令。
- 可评估双缓冲和一次处理多行。
- mask 对每个核只需加载一次，当前已避免逐行重复 GM 搬运。

### LDPC Recover

已完成：

- 独立码块多核并行。
- 消息数组和硬判决放在 UB。
- syndrome 为 0 时提前结束。
- 迭代次数可观测。

主要瓶颈：

- Tanner 图稀疏、不规则访问。
- CN/VN 更新存在轮间依赖。
- 图索引当前从 GM 读取。
- 520×uint16 不是整 32-byte DMA 长度，当前安全标量搬入。
- 一个码块内部首版未跨核并行。

可能的后续优化：

- 使用 `DataCopyPad` 或显式 padding。
- 将固定图索引分块缓存到 UB。
- 研究 layered schedule，减少迭代次数。
- 使用更适合 Vector Core 的消息布局。
- 量化 LLR/message 到 float16 或定点，但必须重新验证精度。
- 根据 batch 和平均迭代动态评估 blocks/core。

## 4. 真机性能测试矩阵

至少覆盖：

| 项目 | 建议取值 |
|---|---|
| Ada batch B | 1、8、20、64、1024、12000 |
| LDPC blocks S | 1、3、20、60、1024、18000 |
| 错误强度 | 0、低、中、高、不可纠正 |
| max_iterations | 10、25、50 |
| 核数策略 | 单核、部分核心、全部可用核心 |
| 场景 | 冷启动、热启动、连续多次 |

## 5. 应报告的指标

AdaMTrans：

- 总时延和单模板时延。
- templates/s。
- GM 读写带宽。
- Vector 利用率。
- 不同 batch 的扩展性。

LDPC：

- 单块和批量时延。
- blocks/s。
- 平均、P95、最大迭代数。
- 每轮平均耗时。
- AI Core/Vector 利用率。
- GM 带宽和 UB 使用量。
- 可纠正与不可纠正输入的时延差异。

端到端：

- H2D、两个算子、D2H 和 Host SHA 分项耗时。
- 总注册/验证吞吐量。
- 首次加载与稳定运行差异。

## 6. 性能测试注意事项

1. 使用事件或设备侧计时，避免只用 Host wall-clock。
2. 预热后再统计。
3. 同步点必须明确。
4. 分开报告数据搬运和 kernel 时间。
5. 记录 SoC、CANN、固件、驱动和频率状态。
6. 报告输入错误分布，因为 LDPC 迭代次数直接影响延迟。
7. 不能把 CPU Debug 时间写成 NPU 性能。
8. 优化后必须与 Host 真值逐 bit 回归，不能只看速度。

## 7. 性能验收建议

当前没有目标设备和业务时延要求，因此不人为设定“必须达到多少毫秒”。在目标
设备上先建立可重复基线，再依据应用吞吐量、FMR/FNMR 和延迟要求决定是否：

- 继续优化 AI Core LDPC；
- 将部分 LDPC 留在高性能 CPU；
- 更换更适合硬件的纠错结构；
- 调整 batch 和异步流水。
