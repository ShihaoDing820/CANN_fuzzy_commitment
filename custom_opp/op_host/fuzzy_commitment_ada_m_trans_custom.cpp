#include <algorithm>
#include <cstdint>

#include "../op_kernel/fuzzy_commitment_ada_m_trans_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kEmbeddingDimension = 512;
constexpr int64_t kBitsPerFeature = 3;
constexpr int64_t kTemplateBits = kEmbeddingDimension * kBitsPerFeature;

bool IsKnownOrExpected(int64_t value, int64_t expected)
{
    return value < 0 || value == expected;
}

bool IsVector(const gert::Shape* shape, int64_t length)
{
    return shape != nullptr && shape->GetDimNum() == 1U &&
        IsKnownOrExpected(shape->GetDim(0), length);
}

uint32_t GetAvailableCoreNum(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return 1U;
    }
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t coreNum = platform.GetCoreNum();
    return coreNum == 0U ? 1U : coreNum;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* embeddingsStorage = context->GetInputShape(0);
    const auto* thresholdsStorage = context->GetInputShape(1);
    const auto* maskStorage = context->GetInputShape(2);
    if (embeddingsStorage == nullptr || thresholdsStorage == nullptr || maskStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& embeddings = embeddingsStorage->GetStorageShape();
    const auto& thresholds = thresholdsStorage->GetStorageShape();
    const auto& mask = maskStorage->GetStorageShape();
    if (embeddings.GetDimNum() != 2U || embeddings.GetDim(1) != kEmbeddingDimension ||
        !IsVector(&thresholds, 3) || !IsVector(&mask, kTemplateBits)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batchValue = embeddings.GetDim(0);
    if (batchValue <= 0 || batchValue > static_cast<int64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* embeddingDim = attrs->GetInt(0);
    const int64_t* bitsPerFeature = attrs->GetInt(1);
    if (embeddingDim == nullptr || bitsPerFeature == nullptr ||
        *embeddingDim != kEmbeddingDimension || *bitsPerFeature != kBitsPerFeature) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(batchValue);
    const uint32_t usedCores = std::min(batch, GetAvailableCoreNum(context));
    auto* tiling =
        context->GetTilingData<FuzzyCommitmentAdaMTransCustomTilingData>();
    if (tiling == nullptr || usedCores == 0U) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = batch;
    tiling->baseRowsPerCore = batch / usedCores;
    tiling->extraRowCoreCount = batch % usedCores;
    tiling->usedCoreNum = usedCores;
    tiling->tileRows = 1U;
    tiling->embeddingDim = static_cast<uint32_t>(*embeddingDim);
    tiling->bitsPerFeature = static_cast<uint32_t>(*bitsPerFeature);

    if (context->SetBlockDim(usedCores) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* embeddings = context->GetInputShape(0);
    const gert::Shape* thresholds = context->GetInputShape(1);
    const gert::Shape* mask = context->GetInputShape(2);
    gert::Shape* output = context->GetOutputShape(0);
    if (embeddings == nullptr || output == nullptr || embeddings->GetDimNum() != 2U ||
        !IsKnownOrExpected(embeddings->GetDim(1), kEmbeddingDimension) ||
        !IsVector(thresholds, 3) || !IsVector(mask, kTemplateBits)) {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape({embeddings->GetDim(0), kTemplateBits});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_UINT16);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FuzzyCommitmentAdaMTransCustom : public OpDef {
public:
    explicit FuzzyCommitmentAdaMTransCustom(const char* name) : OpDef(name)
    {
        this->Input("embeddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("thresholds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("binary_templates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("embedding_dim").AttrType(OPTIONAL).Int(kEmbeddingDimension);
        this->Attr("bits_per_feature").AttrType(OPTIONAL).Int(kBitsPerFeature);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(FuzzyCommitmentAdaMTransCustom);

}  // namespace ops
