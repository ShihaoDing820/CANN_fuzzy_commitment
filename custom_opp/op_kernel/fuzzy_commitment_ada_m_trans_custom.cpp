#include "kernel_operator.h"
#include "fuzzy_commitment_ada_m_trans_custom_tiling.h"

namespace {

constexpr uint32_t kEmbeddingDimension = 512;
constexpr uint32_t kBitsPerFeature = 3;
constexpr uint32_t kTemplateBits = kEmbeddingDimension * kBitsPerFeature;
constexpr uint32_t kBufferNum = 1;

class FuzzyCommitmentAdaMTransKernel {
public:
    __aicore__ inline FuzzyCommitmentAdaMTransKernel() = default;

    __aicore__ inline void Init(
        GM_ADDR embeddings,
        GM_ADDR thresholds,
        GM_ADDR mask,
        GM_ADDR binaryTemplates,
        const FuzzyCommitmentAdaMTransCustomTilingData& tiling)
    {
        embeddingsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(embeddings),
            tiling.batchSize * kEmbeddingDimension);
        thresholdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(thresholds), 3);
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(mask), kTemplateBits);
        binaryTemplatesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint16_t*>(binaryTemplates),
            tiling.batchSize * kTemplateBits);

        const uint32_t core = AscendC::GetBlockIdx();
        rowCount_ = tiling.baseRowsPerCore +
            (core < tiling.extraRowCoreCount ? 1U : 0U);
        startRow_ = core * tiling.baseRowsPerCore +
            (core < tiling.extraRowCoreCount ? core : tiling.extraRowCoreCount);

        pipe_.InitBuffer(embeddingQueue_, kBufferNum, kEmbeddingDimension * sizeof(float));
        pipe_.InitBuffer(outputQueue_, kBufferNum, kTemplateBits * sizeof(uint16_t));
        pipe_.InitBuffer(maskBuffer_, kTemplateBits * sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<uint16_t> maskLocal = maskBuffer_.Get<uint16_t>();
        AscendC::DataCopy(maskLocal, maskGm_[0], kTemplateBits);

        for (uint32_t localRow = 0; localRow < rowCount_; ++localRow) {
            const uint32_t row = startRow_ + localRow;
            CopyIn(row);
            Compute(maskLocal);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        AscendC::LocalTensor<float> embeddingLocal = embeddingQueue_.AllocTensor<float>();
        AscendC::DataCopy(
            embeddingLocal,
            embeddingsGm_[row * kEmbeddingDimension],
            kEmbeddingDimension);
        embeddingQueue_.EnQue(embeddingLocal);
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<uint16_t>& maskLocal)
    {
        AscendC::LocalTensor<float> embeddingLocal = embeddingQueue_.DeQue<float>();
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue_.AllocTensor<uint16_t>();

        const float threshold0 = thresholdsGm_.GetValue(0);
        const float threshold1 = thresholdsGm_.GetValue(1);
        const float threshold2 = thresholdsGm_.GetValue(2);
        for (uint32_t feature = 0; feature < kEmbeddingDimension; ++feature) {
            const float value = embeddingLocal.GetValue(feature);
            uint32_t level = 0;
            level += value >= threshold0 ? 1U : 0U;
            level += value >= threshold1 ? 1U : 0U;
            level += value >= threshold2 ? 1U : 0U;
            const uint32_t base = feature * kBitsPerFeature;
            outputLocal.SetValue(
                base,
                static_cast<uint16_t>(level >= 3U ? 1U : 0U) & maskLocal.GetValue(base));
            outputLocal.SetValue(
                base + 1U,
                static_cast<uint16_t>(level >= 2U ? 1U : 0U) & maskLocal.GetValue(base + 1U));
            outputLocal.SetValue(
                base + 2U,
                static_cast<uint16_t>(level >= 1U ? 1U : 0U) & maskLocal.GetValue(base + 2U));
        }

        outputQueue_.EnQue(outputLocal);
        embeddingQueue_.FreeTensor(embeddingLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<uint16_t> outputLocal = outputQueue_.DeQue<uint16_t>();
        AscendC::DataCopy(
            binaryTemplatesGm_[row * kTemplateBits],
            outputLocal,
            kTemplateBits);
        outputQueue_.FreeTensor(outputLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kBufferNum> embeddingQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBufferNum> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuffer_;
    AscendC::GlobalTensor<float> embeddingsGm_;
    AscendC::GlobalTensor<float> thresholdsGm_;
    AscendC::GlobalTensor<uint16_t> maskGm_;
    AscendC::GlobalTensor<uint16_t> binaryTemplatesGm_;
    uint32_t startRow_ = 0;
    uint32_t rowCount_ = 0;
};

}  // namespace

#if defined(ASCENDC_CPU_DEBUG) || defined(ASCENDC_DEBUG)
#define FUZZY_COMMITMENT_KERNEL_TYPE __vector__
#else
#define FUZZY_COMMITMENT_KERNEL_TYPE __aicore__
#endif

extern "C" __global__ FUZZY_COMMITMENT_KERNEL_TYPE void fuzzy_commitment_ada_m_trans_custom(
    GM_ADDR embeddings,
    GM_ADDR thresholds,
    GM_ADDR mask,
    GM_ADDR binaryTemplates,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FuzzyCommitmentAdaMTransCustomTilingData);
    FuzzyCommitmentAdaMTransKernel kernel;
#if defined(ASCENDC_CPU_DEBUG) || defined(ASCENDC_DEBUG)
    const auto* tilingGm =
        reinterpret_cast<__gm__ FuzzyCommitmentAdaMTransCustomTilingData*>(tiling);
    FuzzyCommitmentAdaMTransCustomTilingData tilingData{
        tilingGm->batchSize,
        tilingGm->baseRowsPerCore,
        tilingGm->extraRowCoreCount,
        tilingGm->usedCoreNum,
        tilingGm->tileRows,
        tilingGm->embeddingDim,
        tilingGm->bitsPerFeature,
    };
    kernel.Init(embeddings, thresholds, mask, binaryTemplates, tilingData);
#else
    GET_TILING_DATA(tilingData, tiling);
    kernel.Init(embeddings, thresholds, mask, binaryTemplates, tilingData);
#endif
    kernel.Process();
}

#undef FUZZY_COMMITMENT_KERNEL_TYPE
