
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FUZZY_COMMITMENT_LDPC_RECOVER_CUSTOM_H_
#define ACLNN_FUZZY_COMMITMENT_LDPC_RECOVER_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFuzzyCommitmentLdpcRecoverCustomGetWorkspaceSize
 * parameters :
 * probe : required
 * helper : required
 * checkOffsets : required
 * checkVariables : required
 * variableOffsets : required
 * variableEdges : required
 * maxIterations : optional
 * channelLlr : optional
 * normalization : optional
 * decodedOut : required
 * decodeOkOut : required
 * syndromeWeightOut : required
 * iterationsOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFuzzyCommitmentLdpcRecoverCustomGetWorkspaceSize(
    const aclTensor *probe,
    const aclTensor *helper,
    const aclTensor *checkOffsets,
    const aclTensor *checkVariables,
    const aclTensor *variableOffsets,
    const aclTensor *variableEdges,
    int64_t maxIterations,
    double channelLlr,
    double normalization,
    const aclTensor *decodedOut,
    const aclTensor *decodeOkOut,
    const aclTensor *syndromeWeightOut,
    const aclTensor *iterationsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFuzzyCommitmentLdpcRecoverCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFuzzyCommitmentLdpcRecoverCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
