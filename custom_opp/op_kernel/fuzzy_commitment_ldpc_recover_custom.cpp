#include "kernel_operator.h"
#include "fuzzy_commitment_ldpc_recover_custom_tiling.h"

namespace {

constexpr uint32_t kCheckCount = 420;
constexpr uint32_t kVariableCount = 520;
constexpr uint32_t kEdgeCount = 1970;

class FuzzyCommitmentLdpcRecoverKernel {
public:
    __aicore__ inline FuzzyCommitmentLdpcRecoverKernel() = default;

    __aicore__ inline void Init(
        GM_ADDR probe,
        GM_ADDR helper,
        GM_ADDR checkOffsets,
        GM_ADDR checkVariables,
        GM_ADDR variableOffsets,
        GM_ADDR variableEdges,
        GM_ADDR decoded,
        GM_ADDR decodeOk,
        GM_ADDR syndromeWeight,
        GM_ADDR iterations,
        const FuzzyCommitmentLdpcRecoverCustomTilingData& tiling)
    {
        probeGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint16_t*>(probe),
            tiling.blockCount * kVariableCount);
        helperGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint16_t*>(helper),
            tiling.blockCount * kVariableCount);
        checkOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(checkOffsets), kCheckCount + 1U);
        checkVariablesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(checkVariables), kEdgeCount);
        variableOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(variableOffsets), kVariableCount + 1U);
        variableEdgesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(variableEdges), kEdgeCount);
        decodedGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint16_t*>(decoded),
            tiling.blockCount * kVariableCount);
        decodeOkGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(decodeOk), tiling.blockCount);
        syndromeWeightGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(syndromeWeight), tiling.blockCount);
        iterationsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(iterations), tiling.blockCount);

        const uint32_t core = AscendC::GetBlockIdx();
        blockCount_ = tiling.baseBlocksPerCore +
            (core < tiling.extraBlockCoreCount ? 1U : 0U);
        startBlock_ = core * tiling.baseBlocksPerCore +
            (core < tiling.extraBlockCoreCount ? core : tiling.extraBlockCoreCount);
        maxIterations_ = tiling.maxIterations;
        channelLlrValue_ = tiling.channelLlr;
        normalization_ = tiling.normalization;

        pipe_.InitBuffer(probeBuffer_, kVariableCount * sizeof(uint16_t));
        pipe_.InitBuffer(helperBuffer_, kVariableCount * sizeof(uint16_t));
        pipe_.InitBuffer(channelLlrBuffer_, kVariableCount * sizeof(float));
        pipe_.InitBuffer(qMessageBuffer_, kEdgeCount * sizeof(float));
        pipe_.InitBuffer(rMessageBuffer_, kEdgeCount * sizeof(float));
        pipe_.InitBuffer(hardDecisionBuffer_, kVariableCount * sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<uint16_t> probeLocal = probeBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<uint16_t> helperLocal = helperBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<float> channelLlr = channelLlrBuffer_.Get<float>();
        AscendC::LocalTensor<float> qMessage = qMessageBuffer_.Get<float>();
        AscendC::LocalTensor<float> rMessage = rMessageBuffer_.Get<float>();
        AscendC::LocalTensor<uint16_t> hardDecision = hardDecisionBuffer_.Get<uint16_t>();

        for (uint32_t localBlock = 0; localBlock < blockCount_; ++localBlock) {
            const uint32_t block = startBlock_ + localBlock;
            CopyIn(block, probeLocal, helperLocal);
            int32_t syndromeWeight = static_cast<int32_t>(kCheckCount);
            int32_t completedIterations = 0;
            Compute(
                probeLocal,
                helperLocal,
                channelLlr,
                qMessage,
                rMessage,
                hardDecision,
                syndromeWeight,
                completedIterations);
            CopyOut(block, hardDecision, syndromeWeight, completedIterations);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t block,
        AscendC::LocalTensor<uint16_t>& probeLocal,
        AscendC::LocalTensor<uint16_t>& helperLocal)
    {
        const uint32_t base = block * kVariableCount;
        // 520 uint16 values are not a whole 32-byte DMA multiple. Scalar copy is
        // deliberately used here to avoid over-reading the final eight bytes.
        for (uint32_t variable = 0; variable < kVariableCount; ++variable) {
            probeLocal.SetValue(variable, probeGm_.GetValue(base + variable));
            helperLocal.SetValue(variable, helperGm_.GetValue(base + variable));
        }
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<uint16_t>& probeLocal,
        AscendC::LocalTensor<uint16_t>& helperLocal,
        AscendC::LocalTensor<float>& channelLlr,
        AscendC::LocalTensor<float>& qMessage,
        AscendC::LocalTensor<float>& rMessage,
        AscendC::LocalTensor<uint16_t>& hardDecision,
        int32_t& syndromeWeight,
        int32_t& completedIterations)
    {
        for (uint32_t variable = 0; variable < kVariableCount; ++variable) {
            const uint16_t received =
                probeLocal.GetValue(variable) ^ helperLocal.GetValue(variable);
            const float llr = received == 0U ? channelLlrValue_ : -channelLlrValue_;
            channelLlr.SetValue(variable, llr);
            hardDecision.SetValue(variable, received);
        }
        for (uint32_t edge = 0; edge < kEdgeCount; ++edge) {
            qMessage.SetValue(
                edge,
                channelLlr.GetValue(static_cast<uint32_t>(checkVariablesGm_.GetValue(edge))));
            rMessage.SetValue(edge, 0.0F);
        }

        for (uint32_t iteration = 0; iteration < maxIterations_; ++iteration) {
            CheckNodeUpdate(qMessage, rMessage);
            VariableNodeUpdate(channelLlr, qMessage, rMessage, hardDecision);
            syndromeWeight = ComputeSyndromeWeight(hardDecision);
            completedIterations = static_cast<int32_t>(iteration + 1U);
            if (syndromeWeight == 0) {
                break;
            }
        }
    }

    __aicore__ inline void CheckNodeUpdate(
        AscendC::LocalTensor<float>& qMessage,
        AscendC::LocalTensor<float>& rMessage)
    {
        for (uint32_t check = 0; check < kCheckCount; ++check) {
            const int32_t begin = checkOffsetsGm_.GetValue(check);
            const int32_t end = checkOffsetsGm_.GetValue(check + 1U);
            float minimum = 1.0e30F;
            float secondMinimum = 1.0e30F;
            int32_t minimumEdge = -1;
            int32_t signProduct = 1;
            for (int32_t edge = begin; edge < end; ++edge) {
                const float value = qMessage.GetValue(static_cast<uint32_t>(edge));
                const float magnitude = value < 0.0F ? -value : value;
                signProduct *= value < 0.0F ? -1 : 1;
                if (magnitude < minimum) {
                    secondMinimum = minimum;
                    minimum = magnitude;
                    minimumEdge = edge;
                } else if (magnitude < secondMinimum) {
                    secondMinimum = magnitude;
                }
            }
            for (int32_t edge = begin; edge < end; ++edge) {
                const int32_t ownSign =
                    qMessage.GetValue(static_cast<uint32_t>(edge)) < 0.0F ? -1 : 1;
                const float magnitude = edge == minimumEdge ? secondMinimum : minimum;
                rMessage.SetValue(
                    static_cast<uint32_t>(edge),
                    normalization_ * static_cast<float>(signProduct * ownSign) * magnitude);
            }
        }
    }

    __aicore__ inline void VariableNodeUpdate(
        AscendC::LocalTensor<float>& channelLlr,
        AscendC::LocalTensor<float>& qMessage,
        AscendC::LocalTensor<float>& rMessage,
        AscendC::LocalTensor<uint16_t>& hardDecision)
    {
        for (uint32_t variable = 0; variable < kVariableCount; ++variable) {
            float total = channelLlr.GetValue(variable);
            const int32_t begin = variableOffsetsGm_.GetValue(variable);
            const int32_t end = variableOffsetsGm_.GetValue(variable + 1U);
            for (int32_t index = begin; index < end; ++index) {
                total += rMessage.GetValue(
                    static_cast<uint32_t>(variableEdgesGm_.GetValue(index)));
            }
            hardDecision.SetValue(variable, total < 0.0F ? 1U : 0U);
            for (int32_t index = begin; index < end; ++index) {
                const uint32_t edge =
                    static_cast<uint32_t>(variableEdgesGm_.GetValue(index));
                qMessage.SetValue(edge, total - rMessage.GetValue(edge));
            }
        }
    }

    __aicore__ inline int32_t ComputeSyndromeWeight(
        AscendC::LocalTensor<uint16_t>& hardDecision)
    {
        int32_t weight = 0;
        for (uint32_t check = 0; check < kCheckCount; ++check) {
            uint16_t parity = 0;
            const int32_t begin = checkOffsetsGm_.GetValue(check);
            const int32_t end = checkOffsetsGm_.GetValue(check + 1U);
            for (int32_t edge = begin; edge < end; ++edge) {
                parity ^= hardDecision.GetValue(
                    static_cast<uint32_t>(checkVariablesGm_.GetValue(edge)));
            }
            weight += parity == 0U ? 0 : 1;
        }
        return weight;
    }

    __aicore__ inline void CopyOut(
        uint32_t block,
        AscendC::LocalTensor<uint16_t>& hardDecision,
        int32_t syndromeWeight,
        int32_t completedIterations)
    {
        const uint32_t base = block * kVariableCount;
        for (uint32_t variable = 0; variable < kVariableCount; ++variable) {
            decodedGm_.SetValue(base + variable, hardDecision.GetValue(variable));
        }
        decodeOkGm_.SetValue(block, syndromeWeight == 0 ? 1 : 0);
        syndromeWeightGm_.SetValue(block, syndromeWeight);
        iterationsGm_.SetValue(block, completedIterations);
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> probeBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> helperBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> channelLlrBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qMessageBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rMessageBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hardDecisionBuffer_;
    AscendC::GlobalTensor<uint16_t> probeGm_;
    AscendC::GlobalTensor<uint16_t> helperGm_;
    AscendC::GlobalTensor<int32_t> checkOffsetsGm_;
    AscendC::GlobalTensor<int32_t> checkVariablesGm_;
    AscendC::GlobalTensor<int32_t> variableOffsetsGm_;
    AscendC::GlobalTensor<int32_t> variableEdgesGm_;
    AscendC::GlobalTensor<uint16_t> decodedGm_;
    AscendC::GlobalTensor<int32_t> decodeOkGm_;
    AscendC::GlobalTensor<int32_t> syndromeWeightGm_;
    AscendC::GlobalTensor<int32_t> iterationsGm_;
    uint32_t startBlock_ = 0;
    uint32_t blockCount_ = 0;
    uint32_t maxIterations_ = 50;
    float channelLlrValue_ = 4.0F;
    float normalization_ = 0.75F;
};

}  // namespace

#if defined(ASCENDC_CPU_DEBUG) || defined(ASCENDC_DEBUG)
#define FUZZY_COMMITMENT_KERNEL_TYPE __vector__
#else
#define FUZZY_COMMITMENT_KERNEL_TYPE __aicore__
#endif

extern "C" __global__ FUZZY_COMMITMENT_KERNEL_TYPE void fuzzy_commitment_ldpc_recover_custom(
    GM_ADDR probe,
    GM_ADDR helper,
    GM_ADDR checkOffsets,
    GM_ADDR checkVariables,
    GM_ADDR variableOffsets,
    GM_ADDR variableEdges,
    GM_ADDR decoded,
    GM_ADDR decodeOk,
    GM_ADDR syndromeWeight,
    GM_ADDR iterations,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FuzzyCommitmentLdpcRecoverCustomTilingData);
    FuzzyCommitmentLdpcRecoverKernel kernel;
#if defined(ASCENDC_CPU_DEBUG) || defined(ASCENDC_DEBUG)
    const auto* tilingGm =
        reinterpret_cast<__gm__ FuzzyCommitmentLdpcRecoverCustomTilingData*>(tiling);
    FuzzyCommitmentLdpcRecoverCustomTilingData tilingData{
        tilingGm->blockCount,
        tilingGm->baseBlocksPerCore,
        tilingGm->extraBlockCoreCount,
        tilingGm->usedCoreNum,
        tilingGm->maxIterations,
        tilingGm->channelLlr,
        tilingGm->normalization,
    };
    kernel.Init(
        probe,
        helper,
        checkOffsets,
        checkVariables,
        variableOffsets,
        variableEdges,
        decoded,
        decodeOk,
        syndromeWeight,
        iterations,
        tilingData);
#else
    GET_TILING_DATA(tilingData, tiling);
    kernel.Init(
        probe,
        helper,
        checkOffsets,
        checkVariables,
        variableOffsets,
        variableEdges,
        decoded,
        decodeOk,
        syndromeWeight,
        iterations,
        tilingData);
#endif
    kernel.Process();
}

#undef FUZZY_COMMITMENT_KERNEL_TYPE
