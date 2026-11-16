
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FUZZY_COMMITMENT_ADA_MTRANS_CUSTOM_H_
#define ACLNN_FUZZY_COMMITMENT_ADA_MTRANS_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFuzzyCommitmentAdaMTransCustomGetWorkspaceSize
 * parameters :
 * embeddings : required
 * thresholds : required
 * mask : required
 * embeddingDim : optional
 * bitsPerFeature : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFuzzyCommitmentAdaMTransCustomGetWorkspaceSize(
    const aclTensor *embeddings,
    const aclTensor *thresholds,
    const aclTensor *mask,
    int64_t embeddingDim,
    int64_t bitsPerFeature,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFuzzyCommitmentAdaMTransCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFuzzyCommitmentAdaMTransCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
