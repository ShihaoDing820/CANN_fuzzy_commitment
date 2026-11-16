#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FuzzyCommitmentAdaMTransCustom)
    .INPUT(embeddings, ge::TensorType::ALL())
    .INPUT(thresholds, ge::TensorType::ALL())
    .INPUT(mask, ge::TensorType::ALL())
    .OUTPUT(binary_templates, ge::TensorType::ALL())
    .ATTR(embedding_dim, Int, 512)
    .ATTR(bits_per_feature, Int, 3)
    .OP_END_FACTORY_REG(FuzzyCommitmentAdaMTransCustom);

REG_OP(FuzzyCommitmentLdpcRecoverCustom)
    .INPUT(probe, ge::TensorType::ALL())
    .INPUT(helper, ge::TensorType::ALL())
    .INPUT(check_offsets, ge::TensorType::ALL())
    .INPUT(check_variables, ge::TensorType::ALL())
    .INPUT(variable_offsets, ge::TensorType::ALL())
    .INPUT(variable_edges, ge::TensorType::ALL())
    .OUTPUT(decoded, ge::TensorType::ALL())
    .OUTPUT(decode_ok, ge::TensorType::ALL())
    .OUTPUT(syndrome_weight, ge::TensorType::ALL())
    .OUTPUT(iterations, ge::TensorType::ALL())
    .ATTR(max_iterations, Int, 50)
    .ATTR(channel_llr, Float, 4)
    .ATTR(normalization, Float, 0.75)
    .OP_END_FACTORY_REG(FuzzyCommitmentLdpcRecoverCustom);

}

#endif
