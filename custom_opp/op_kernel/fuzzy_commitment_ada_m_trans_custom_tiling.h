#ifndef FUZZY_COMMITMENT_ADA_M_TRANS_CUSTOM_TILING_H
#define FUZZY_COMMITMENT_ADA_M_TRANS_CUSTOM_TILING_H

#include <cstdint>

// Runtime tiling data produced on the Host and consumed by the AI Core kernel.
// Rows are distributed as baseRowsPerCore plus one extra row on the first
// extraRowCoreCount cores.
struct FuzzyCommitmentAdaMTransCustomTilingData {
    uint32_t batchSize;
    uint32_t baseRowsPerCore;
    uint32_t extraRowCoreCount;
    uint32_t usedCoreNum;
    uint32_t tileRows;
    uint32_t embeddingDim;
    uint32_t bitsPerFeature;
};

#endif  // FUZZY_COMMITMENT_ADA_M_TRANS_CUSTOM_TILING_H
