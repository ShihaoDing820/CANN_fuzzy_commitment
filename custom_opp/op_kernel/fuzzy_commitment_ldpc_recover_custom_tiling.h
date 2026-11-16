#ifndef FUZZY_COMMITMENT_LDPC_RECOVER_CUSTOM_TILING_H
#define FUZZY_COMMITMENT_LDPC_RECOVER_CUSTOM_TILING_H

#include <cstdint>

// A complete 520-bit code block stays on one core because the CN/VN message
// updates have iteration-wide dependencies. Independent blocks are parallel.
struct FuzzyCommitmentLdpcRecoverCustomTilingData {
    uint32_t blockCount;
    uint32_t baseBlocksPerCore;
    uint32_t extraBlockCoreCount;
    uint32_t usedCoreNum;
    uint32_t maxIterations;
    float channelLlr;
    float normalization;
};

#endif  // FUZZY_COMMITMENT_LDPC_RECOVER_CUSTOM_TILING_H
