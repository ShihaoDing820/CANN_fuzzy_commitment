#include <algorithm>
#include <cstdint>

#include "../op_kernel/fuzzy_commitment_ldpc_recover_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kVariableCount = 520;
constexpr int64_t kCheckCount = 420;
constexpr int64_t kEdgeCount = 1970;

bool IsKnownOrExpected(int64_t value, int64_t expected)
{
    return value < 0 || value == expected;
}

bool IsVector(const gert::Shape* shape, int64_t length)
{
    return shape != nullptr && shape->GetDimNum() == 1U &&
        IsKnownOrExpected(shape->GetDim(0), length);
}

bool IsExactStorageVector(const gert::StorageShape* storage, int64_t length)
{
    if (storage == nullptr) {
        return false;
    }
    const auto& shape = storage->GetStorageShape();
    return shape.GetDimNum() == 1U && shape.GetDim(0) == length;
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
    const auto* probeStorage = context->GetInputShape(0);
    const auto* helperStorage = context->GetInputShape(1);
    if (probeStorage == nullptr || helperStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& probe = probeStorage->GetStorageShape();
    const auto& helper = helperStorage->GetStorageShape();
    if (probe.GetDimNum() != 2U || helper.GetDimNum() != 2U ||
        probe.GetDim(1) != kVariableCount || helper.GetDim(1) != kVariableCount ||
        probe.GetDim(0) != helper.GetDim(0) || probe.GetDim(0) <= 0 ||
        !IsExactStorageVector(context->GetInputShape(2), kCheckCount + 1) ||
        !IsExactStorageVector(context->GetInputShape(3), kEdgeCount) ||
        !IsExactStorageVector(context->GetInputShape(4), kVariableCount + 1) ||
        !IsExactStorageVector(context->GetInputShape(5), kEdgeCount)) {
        return ge::GRAPH_FAILED;
    }
    if (probe.GetDim(0) > static_cast<int64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* maxIterations = attrs->GetInt(0);
    const float* channelLlr = attrs->GetFloat(1);
    const float* normalization = attrs->GetFloat(2);
    if (maxIterations == nullptr || channelLlr == nullptr || normalization == nullptr ||
        *maxIterations <= 0 || *maxIterations > 1000 || *channelLlr <= 0.0F ||
        *normalization <= 0.0F || *normalization > 1.0F) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockCount = static_cast<uint32_t>(probe.GetDim(0));
    const uint32_t usedCores = std::min(blockCount, GetAvailableCoreNum(context));
    auto* tiling =
        context->GetTilingData<FuzzyCommitmentLdpcRecoverCustomTilingData>();
    if (tiling == nullptr || usedCores == 0U) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockCount = blockCount;
    tiling->baseBlocksPerCore = blockCount / usedCores;
    tiling->extraBlockCoreCount = blockCount % usedCores;
    tiling->usedCoreNum = usedCores;
    tiling->maxIterations = static_cast<uint32_t>(*maxIterations);
    tiling->channelLlr = *channelLlr;
    tiling->normalization = *normalization;

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
    const gert::Shape* probe = context->GetInputShape(0);
    const gert::Shape* helper = context->GetInputShape(1);
    gert::Shape* decoded = context->GetOutputShape(0);
    gert::Shape* decodeOk = context->GetOutputShape(1);
    gert::Shape* syndromeWeight = context->GetOutputShape(2);
    gert::Shape* iterations = context->GetOutputShape(3);
    if (probe == nullptr || helper == nullptr || decoded == nullptr || decodeOk == nullptr ||
        syndromeWeight == nullptr || iterations == nullptr ||
        probe->GetDimNum() != 2U || helper->GetDimNum() != 2U ||
        !IsKnownOrExpected(probe->GetDim(1), kVariableCount) ||
        !IsKnownOrExpected(helper->GetDim(1), kVariableCount) ||
        (probe->GetDim(0) >= 0 && helper->GetDim(0) >= 0 &&
         probe->GetDim(0) != helper->GetDim(0)) ||
        !IsVector(context->GetInputShape(2), kCheckCount + 1) ||
        !IsVector(context->GetInputShape(3), kEdgeCount) ||
        !IsVector(context->GetInputShape(4), kVariableCount + 1) ||
        !IsVector(context->GetInputShape(5), kEdgeCount)) {
        return ge::GRAPH_FAILED;
    }

    *decoded = gert::Shape({probe->GetDim(0), kVariableCount});
    const gert::Shape metadataShape({probe->GetDim(0)});
    *decodeOk = metadataShape;
    *syndromeWeight = metadataShape;
    *iterations = metadataShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_UINT16);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FuzzyCommitmentLdpcRecoverCustom : public OpDef {
public:
    explicit FuzzyCommitmentLdpcRecoverCustom(const char* name) : OpDef(name)
    {
        this->Input("probe")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("helper")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("check_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("check_variables")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("variable_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("variable_edges")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("decoded")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("decode_ok")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("syndrome_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("iterations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("max_iterations").AttrType(OPTIONAL).Int(50);
        this->Attr("channel_llr").AttrType(OPTIONAL).Float(4.0F);
        this->Attr("normalization").AttrType(OPTIONAL).Float(0.75F);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(FuzzyCommitmentLdpcRecoverCustom);

}  // namespace ops
