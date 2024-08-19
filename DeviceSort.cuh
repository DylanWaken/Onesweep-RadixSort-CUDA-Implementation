//
// Created by dylan on 13/08/24.
//

#ifndef CUDATESTS_DEVICESORT_CUH
#define CUDATESTS_DEVICESORT_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "cudaMacros.cuh"
#include "DeviceScan.cuh"


#define DEFAULT_RADIX               256     //The range of digits, 0-255, each uint32_t key takes 4 digits
#define DEFAULT_RADIX_LOG           8       // the log of the radix
// we consider each digit as a bin, and each byte as a digit
#define DEFAULT_HIST_SUB_BLOCKS 4          // the number of stat blocks running in parallel to accelerate atomic operations
#define DEFAULT_HIST_PART_SIZE 16384       // number of elements processed by each block, higher to avoid atomic operations on global memory
#define DEFAULT_HIST_BLOCK_SIZE 256

template<typename T,
        const uint32_t HIST_PART_SIZE,
        const uint32_t HIST_SUB_BLOCKS,
        const uint32_t BLOCK_SIZE,
        const uint32_t RADIX ,
        const uint32_t RADIX_LOG,
        const uint32_t RADIX_MASK
        >
__global__ void globalByteHistogramImpl(T *g_ikeys, uint32_t *g_globalHistogram, uint32_t n) {
    const uint32_t numDigits = sizeof(T);

    // declare shared memory
    __shared__ uint32_t s_globalHistBlock[numDigits][RADIX * HIST_SUB_BLOCKS];

    // reset shared memory
    #pragma unroll
    for (uint32_t j = threadIdx.x; j < RADIX * HIST_SUB_BLOCKS; j += blockDim.x) {
        #pragma unroll numDigits
        for (uint32_t i = 0; i < numDigits; i++) {
            s_globalHistBlock[i][j] = 0;
        }
    }

    __syncthreads();

    // compute thread histogram
    // fetch size allows us to read 128 bits at a time, using vectorized loads
    const uint32_t histogramFetchIndex = (threadIdx.x / (blockDim.x / HIST_SUB_BLOCKS)) * RADIX;

    // compute thread histogram, for blocks not on the boundary we can use vectorized loads
    const uint32_t partEnd = (blockIdx.x + 1) * HIST_PART_SIZE > n ? n : (blockIdx.x + 1) * HIST_PART_SIZE;
    #pragma unroll
    for (uint32_t i = threadIdx.x + (blockIdx.x * HIST_PART_SIZE); i < partEnd; i += blockDim.x) {
        T key = g_ikeys[i];
        #pragma unroll numDigits
        for (uint32_t j = 0; j < numDigits; j++) {
            const uint32_t digit = (key >> (j * RADIX_LOG)) & RADIX_MASK;
            atomicAdd(&s_globalHistBlock[j][histogramFetchIndex + digit], 1);
        }
    }

    __syncthreads();

    // reduce the histograms in blocks
    #pragma unroll
    for (uint32_t k = threadIdx.x; k < RADIX; k += blockDim.x) {
        #pragma unroll
        for (uint32_t j = 1; j < HIST_SUB_BLOCKS; j++) {
            #pragma unroll numDigits
            for (uint32_t i = 0; i < numDigits; i++) {
                s_globalHistBlock[i][k] += s_globalHistBlock[i][k + j * RADIX];
            }
        }
    }

    __syncthreads();

    // write the global histogram
    #pragma unroll
    for (uint32_t j = 0; j < RADIX; j++) {
        #pragma unroll numDigits
        for (uint32_t i = threadIdx.x; i < numDigits; i += blockDim.x) {
            atomicAdd(&g_globalHistogram[i * RADIX + j], s_globalHistBlock[i][j]);
        }
    }
}

#define RANK_MASK 0xff
#define MAX_RANK_MASK 0xff00
#define MAX_RANK_SHIFT 8

#define BIN_HIST_MASK 0xffff
#define BIN_EXCL_MASK 0xffff0000
#define BIN_EXCL_SHIFT 16

template<typename TKey, const uint32_t RADIX_LOG, const uint32_t RADIX_MASK>
__forceinline__ __device__ void warpLevelMultiSplit(TKey val, uint32_t* s_warpHistogram, uint32_t activeDigit, uint32_t& threadRank) {
    uint32_t digitVal = (val >> (activeDigit * RADIX_LOG)) & RADIX_MASK;

    // Perform warp-wide voting for each bit in the digit
    uint32_t digitPopulation = 0xFFFFFFFF;  // Start with all bits set
    #pragma unroll RADIX_LOG
    for (uint32_t bit = 0; bit < RADIX_LOG; ++bit) {
        uint32_t bitMask = 1 << bit;
        uint32_t vote = __ballot_sync(__activemask(), (digitVal & bitMask) != 0);
        digitPopulation &= (digitVal & bitMask) != 0 ? vote : ~vote;
    }

    // Count the number of threads with the same digit
    uint32_t digitCount = __popc(digitPopulation);

    // Determine the rank of the current thread within its digit population : & 31 equivalent to % 32
    // 1 << (threadIdx.x & 31) creates a mask where only the bit corresponding to the current thread's position is set.
    // (1 << (threadIdx.x & 31)) - 1) Subtracting 1 from the previous result creates a mask where all bits up to (but not including) the current thread's position are set.
    // because the thread votes the bite of its rank in `digitPopulation` counting from LSB,
    // & with mask of threadId - 1 gives the rank, and the lowest rank thread has rank 0
    // BIG BRAIN WTF
    uint32_t digitRank = __popc(digitPopulation & ((1 << (threadIdx.x & 31)) - 1));

    // The lowest ranked thread in each digit population updates the histogram
    if (digitRank == 0) {
        atomicAdd(&s_warpHistogram[digitVal], digitCount);
    }

    uint32_t maxRank = digitCount;
    threadRank = (maxRank << MAX_RANK_SHIFT) | digitRank;
}


// when size > 32, it will be irrelevant
__forceinline__ __device__ void warpLevelScan(uint32_t* input, uint32_t* output, uint32_t size, bool exclusive,
                                              uint32_t outputShift = 0) {
    if (size == 0) return;
    __syncwarp();
    uint32_t lane_id = threadIdx.x & 31;  // Get the lane ID within the warp
    uint32_t read_id = lane_id;
    uint32_t add_val = 0;
    #pragma unroll
    while (size > 0) {
        uint32_t valueInit = lane_id < size ? (input[read_id] & BIN_HIST_MASK) : 0;
        uint32_t value = valueInit;

        // Hillis Steele Scan
        // https://www.geeksforgeeks.org/hillis-steele-scan-parallel-prefix-scan-algorithm/
        #pragma unroll 5
        for (int offset = 1; offset < 32; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xffffffff, value, offset);
            value += lane_id >= offset ? n : 0;
        }

        // conversion to exclusive
        if (lane_id < size) {
            // concatinate with input if we have outputShift > 0
            output[read_id] = ((add_val + value - (exclusive ? input[read_id] : 0)) << outputShift) | (outputShift
                    > 0 ? valueInit : 0);
        }
        __syncwarp();
        size -= WARP_SIZE;
        read_id += WARP_SIZE;

        // add the last element up
        add_val += __shfl_sync(0xffffffff, value, 31);
    }
}
#define DEFAULT_BINNING_ITEMS_PER_THREAD 16
#define DEFAULT_BINNING_BLOCK_SIZE 256

#define COUNTER_FLAG_MASK 0b11 << 30
#define COUNTER_VALUE_MASK 0x3fffffff
#define COUNTER_FLAG_NOT_READY (0b00 << 30)
#define COUNTER_FLAG_LOCAL_COUNT (0b01 << 30)
#define COUNTER_FLAG_GLOBAL_SUM (0b10 << 30)


template<typename TKeys,
        typename TValIndex,
        const uint32_t ITEMS_PER_THREAD,  // default 32
        const uint32_t BLOCK_SIZE, // THREADS_PER_BLOCK, default 256
        const uint32_t RADIX,
        const uint32_t RADIX_LOG,
        const uint32_t RADIX_MASK
        >
__global__ void globalBinningImpl(
        TKeys *g_ikeys,
        TKeys *g_okeys,
        TValIndex *g_ivalsIndices,
        TValIndex *g_ovalsIndices,
        uint32_t *g_exclusiveCount,          // the global prefix sum
        uint32_t *g_atomicTileAssignCounter, // single value
        uint32_t *g_statusCounter,    // of size RADIX x number of TILES
        uint32_t n,
        uint32_t currentDigit
) {
    // figure out some useful constants
    const uint32_t NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const uint32_t BIN_TILE_SIZE = ITEMS_PER_THREAD * BLOCK_SIZE;

    // active digit for global exclusive scan prefix sum for each bin (digit)
    uint32_t* g_activeExclusiveCount = g_exclusiveCount + currentDigit * RADIX;

    // declare shared memory, for each uint32_t entry, we have [exclusive sum | histogram]
    __shared__ uint32_t s_warpHistograms[NUM_WARPS << RADIX_LOG];
    __shared__ uint32_t s_blockHistogram[RADIX];
    // an array to store the global (digit and tile) offset of each bin (digit)
    __shared__ uint32_t s_globalOffsets[RADIX];
    uint32_t* s_digitPlacementStats = s_warpHistograms; // memory reuse

    // the warp's histogram
    uint32_t* s_warpHist = s_warpHistograms + (WARP_INDEX << RADIX_LOG);

    // declare register array
    // the keys hold by each thread
    // TODO: Change this whenever we change the ITEMS_PER_THREAD
    TKeys threadKey0;
    TKeys threadKey1;
    TKeys threadKey2;
    TKeys threadKey3;
    TKeys threadKey4;
    TKeys threadKey5;
    TKeys threadKey6;
    TKeys threadKey7;
    TKeys threadKey8;
    TKeys threadKey9;
    TKeys threadKey10;
    TKeys threadKey11;
    TKeys threadKey12;
    TKeys threadKey13;
    TKeys threadKey14;
    TKeys threadKey15;

    uint32_t threadRank0;
    uint32_t threadRank1;
    uint32_t threadRank2;
    uint32_t threadRank3;
    uint32_t threadRank4;
    uint32_t threadRank5;
    uint32_t threadRank6;
    uint32_t threadRank7;
    uint32_t threadRank8;
    uint32_t threadRank9;
    uint32_t threadRank10;
    uint32_t threadRank11;
    uint32_t threadRank12;
    uint32_t threadRank13;
    uint32_t threadRank14;
    uint32_t threadRank15;

    //clear shared memory
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < RADIX * NUM_WARPS; i += BLOCK_SIZE) {
        s_warpHistograms[i] = 0;
    }
    s_blockHistogram[threadIdx.x] = 0;
    s_globalOffsets[threadIdx.x] = 0;
    __syncthreads();

    uint32_t tileId = blockIdx.x;
    // initialize counters
    #pragma unroll
    for(uint32_t i = threadIdx.x; i < RADIX; i+= BLOCK_SIZE){
        g_statusCounter[gridDim.x * i + tileId] = COUNTER_FLAG_NOT_READY;
    }


    // read in elements
    uint32_t itemIndex = tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + LANE_INDEX;
    threadKey0 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey1 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey2 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey3 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey4 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey5 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey6 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey7 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey8 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey9 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey10 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey11 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey12 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey13 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey14 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    itemIndex += WARP_SIZE;
    threadKey15 = itemIndex < n ? g_ikeys[itemIndex] : (~((TKeys)0));
    
    __syncwarp();

    // perform warp level statistics
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey0, s_warpHist, currentDigit, threadRank0);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey1, s_warpHist, currentDigit, threadRank1);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey2, s_warpHist, currentDigit, threadRank2);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey3, s_warpHist, currentDigit, threadRank3);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey4, s_warpHist, currentDigit, threadRank4);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey5, s_warpHist, currentDigit, threadRank5);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey6, s_warpHist, currentDigit, threadRank6);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey7, s_warpHist, currentDigit, threadRank7);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey8, s_warpHist, currentDigit, threadRank8);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey9, s_warpHist, currentDigit, threadRank9);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey10, s_warpHist, currentDigit, threadRank10);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey11, s_warpHist, currentDigit, threadRank11);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey12, s_warpHist, currentDigit, threadRank12);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey13, s_warpHist, currentDigit, threadRank13);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey14, s_warpHist, currentDigit, threadRank14);
    warpLevelMultiSplit<TKeys, RADIX_LOG, RADIX_MASK>(threadKey15, s_warpHist, currentDigit, threadRank15);

    __syncthreads();

    // do a prefix sum for each warp's histograms
    warpLevelScan(s_warpHist, s_warpHist, RADIX, true, BIN_EXCL_SHIFT);

    __syncthreads();
    // do a collection of warp conclusions into a single histogram
    #pragma unroll
    for (uint32_t j = threadIdx.x; j < RADIX; j += BLOCK_SIZE) {
        #pragma unroll NUM_WARPS
        for (uint32_t i = 0; i < NUM_WARPS; i++) {
            // we add both histograms and exclusive sums in this way
            s_blockHistogram[j] += s_warpHistograms[RADIX * i + j];
        }
    }

    __syncthreads();

    // update offsets for each warp group
    #pragma unroll
    for (uint32_t j = threadIdx.x; j < RADIX; j += BLOCK_SIZE) {
        uint32_t runningSum = 0;
        #pragma unroll NUM_WARPS
        for (uint32_t i = 0; i < NUM_WARPS; i++) {
           uint32_t val = s_warpHistograms[RADIX * i + j] & BIN_HIST_MASK;
           // assign runningSum to the the exclusive sum components
           s_warpHistograms[RADIX * i + j] = (runningSum << BIN_EXCL_SHIFT) | val;
           runningSum += val;
        }
    }

    __syncthreads();

    // report the local counters
    #pragma unroll
    for (uint32_t j = threadIdx.x; j < RADIX; j += BLOCK_SIZE) {
        g_statusCounter[j * gridDim.x + tileId] =  COUNTER_FLAG_LOCAL_COUNT | (s_blockHistogram[j] & BIN_HIST_MASK);
    }
    __syncthreads();

    // Perform chained scan prefix sum for each digit
    #pragma unroll
    for (uint32_t digit = threadIdx.x; digit < RADIX; digit += BLOCK_SIZE) {
        uint32_t localCount = (s_blockHistogram[digit] & BIN_HIST_MASK);
        uint32_t exclusivePrefix = 0;

        // Scan backwards through tiles
        for (int32_t prevTile = tileId - 1; prevTile >= 0; --prevTile) {
            uint32_t prevCounter;
            do {
                prevCounter = g_statusCounter[digit * gridDim.x + prevTile];

                // Wait if the previous tile's count is not ready
                if ((prevCounter & COUNTER_FLAG_MASK) == COUNTER_FLAG_NOT_READY) {
                    __threadfence_block();
                }
            } while ((prevCounter & COUNTER_FLAG_MASK) == COUNTER_FLAG_NOT_READY);

            if ((prevCounter & COUNTER_FLAG_MASK) == COUNTER_FLAG_GLOBAL_SUM) {
                // Found a global count, use it and stop scanning
                exclusivePrefix += prevCounter & COUNTER_VALUE_MASK;
                break;
            } else {
                // Add local count to our exclusive prefix
                exclusivePrefix += prevCounter & COUNTER_VALUE_MASK;
            }
        }

        // Calculate inclusive prefix and update status counter
        g_statusCounter[digit * gridDim.x + tileId] = COUNTER_FLAG_GLOBAL_SUM | (exclusivePrefix + localCount);
        // the tile's bin offset = the global digit offset + global tile's offset
        s_globalOffsets[digit] = exclusivePrefix + g_activeExclusiveCount[digit];
    }

    __syncthreads();

    #pragma unroll
    for (uint32_t i = threadIdx.x; i < RADIX * NUM_WARPS; i += BLOCK_SIZE) {
        // remove the lower bits
        s_digitPlacementStats[i] &= BIN_EXCL_MASK;
    }
    __syncthreads();

    // scatter the keys to the bins
    // block 0
    {
        uint32_t digit = (threadKey0 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        // total offset = global digit offset + warp's offset + thread's offset
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank0 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            // scatter the key to the bin
            g_okeys[completeOffset] = threadKey0;

            // scatter the index into the output array
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 0 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        // if we are the lowest ranked thread in the warp, we need to update the global offset
        if (completeOffset < n && (threadRank0 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank0 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 1
    {
        uint32_t digit = (threadKey1 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        // total offset = global digit offset + warp's offset + thread's offset
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank1 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            // scatter the key to the bin
            g_okeys[completeOffset] = threadKey1;

            // scatter the index into the output array
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 1 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        // if we are the lowest ranked thread in the warp, we need to update the global offset
        if (completeOffset < n && (threadRank1 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank1 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }
    
    // block 2
    {
        uint32_t digit = (threadKey2 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        // total offset = global digit offset + warp's offset + thread's offset
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank2 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            // scatter the key to the bin
            g_okeys[completeOffset] = threadKey2;

            // scatter the index into the output array
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 2 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        // if we are the lowest ranked thread in the warp, we need to update the global offset
        if (completeOffset < n && (threadRank2 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank2 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 3
    {
        uint32_t digit = (threadKey3 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank3 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey3;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 3 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank3 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank3 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 4
    {
        uint32_t digit = (threadKey4 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank4 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey4;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 4 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank4 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank4 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 5
    {
        uint32_t digit = (threadKey5 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank5 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey5;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 5 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank5 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank5 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 6
    {
        uint32_t digit = (threadKey6 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank6 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey6;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 6 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank6 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank6 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 7
    {
        uint32_t digit = (threadKey7 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank7 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey7;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 7 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank7 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank7 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 8
    {
        uint32_t digit = (threadKey8 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank8 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey8;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 8 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank8 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank8 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 9
    {
        uint32_t digit = (threadKey9 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank9 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey9;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 9 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank9 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank9 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 10
    {
        uint32_t digit = (threadKey10 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank10 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey10;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 10 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank10 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank10 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 11
    {
        uint32_t digit = (threadKey11 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank11 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey11;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 11 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank11 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank11 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 12
    {
        uint32_t digit = (threadKey12 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank12 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey12;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 12 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank12 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank12 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 13
    {
        uint32_t digit = (threadKey13 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank13 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey13;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 13 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank13 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank13 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 14
    {
        uint32_t digit = (threadKey14 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank14 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey14;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 14 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank14 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank14 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }

    // block 15
    {
        uint32_t digit = (threadKey15 >> (currentDigit * RADIX_LOG)) & RADIX_MASK;
        uint32_t digitPlacementOffset = (s_digitPlacementStats[WARP_INDEX * RADIX + digit] & BIN_HIST_MASK) + (threadRank15 & RANK_MASK);
        uint32_t completeOffset = s_globalOffsets[digit] + ((s_warpHistograms[WARP_INDEX * RADIX + digit] & BIN_EXCL_MASK) >> BIN_EXCL_SHIFT)
            + digitPlacementOffset;

        if (completeOffset < n){
            g_okeys[completeOffset] = threadKey15;
            g_ovalsIndices[completeOffset] = g_ivalsIndices[tileId * BIN_TILE_SIZE + WARP_INDEX * WARP_SIZE * ITEMS_PER_THREAD + 15 * WARP_SIZE + LANE_INDEX];
        }
        __syncwarp();
        if (completeOffset < n && (threadRank15 & RANK_MASK) == 0) {
            s_digitPlacementStats[WARP_INDEX * RADIX + digit] += (threadRank15 & MAX_RANK_MASK) >> MAX_RANK_SHIFT;
        }
        __syncwarp();
    }
}


template<typename T>
__global__ void countExclusiveScanImpl(uint32_t *g_idata, uint32_t *g_odata) {
    extern __shared__ uint32_t temp[];// allocated on invocation

    uint32_t thid = threadIdx.x;
    uint32_t offset = 1;
    uint32_t operative_size = 2 * blockDim.x;

    g_idata += blockIdx.x * operative_size;
    g_odata += blockIdx.x * operative_size;

    // fetch data
    uint32_t ai = thid;
    uint32_t bi = thid + (operative_size / 2);
    uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];           // load input into shared memory without bank conflicts
    temp[bi + bankOffsetB] = g_idata[bi];           // load input into shared memory without bank conflicts
    __syncthreads();

    // build sum in place up the tree
    #pragma unroll
    for (uint32_t d = operative_size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            uint32_t aip = offset * (2 * thid + 1) - 1;
            uint32_t bip = offset * (2 * thid + 2) - 1;
            aip += CONFLICT_FREE_OFFSET(aip);
            bip += CONFLICT_FREE_OFFSET(bip);
            temp[bip] += temp[aip];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[(operative_size - 1) + CONFLICT_FREE_OFFSET(operative_size - 1)] = 0;
    } // clear the last element

    // traverse down tree & build scan
    #pragma unroll
    for (uint32_t d = 1; d < operative_size; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            uint32_t aip = offset * (2 * thid + 1) - 1;
            uint32_t bip = offset * (2 * thid + 2) - 1;
            aip += CONFLICT_FREE_OFFSET(aip);
            bip += CONFLICT_FREE_OFFSET(bip);

            T t = temp[aip];
            temp[aip] = temp[bip];
            temp[bip] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
}

template<typename TIndex>
__global__ void generateIndexArrayImpl(TIndex *g_indices, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g_indices[idx] = idx;
    }
}


// ---------------------------- public interface ----------------------------
// --------------------------------------------------------------------------

template<typename TIndex,
        const uint32_t BLOCK_SIZE = DEFAULT_BINNING_BLOCK_SIZE
        >
inline void generateIndexArray(TIndex *g_indices, uint32_t n) {
    const uint32_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    checkCUDA((generateIndexArrayImpl<TIndex><<<numBlocks, BLOCK_SIZE>>>(g_indices, n)));
}

template<typename T,
        const uint32_t HIST_PART_SIZE,
        const uint32_t HIST_SUB_BLOCKS,
        const uint32_t BLOCK_SIZE,
        const uint32_t RADIX,
        const uint32_t RADIX_LOG,
        const uint32_t RADIX_MASK
>
inline void globalHistogram(T *g_ikeys, uint32_t *g_globalHistogram, uint32_t n) {
    assert(RADIX < WARP_SIZE * WARP_SIZE);
    const uint32_t numBlocks = (n + HIST_PART_SIZE - 1) / HIST_PART_SIZE;
    // run the global histogram
    checkCUDA((globalByteHistogramImpl<T, HIST_PART_SIZE, HIST_SUB_BLOCKS, BLOCK_SIZE, RADIX, RADIX_LOG, RADIX_MASK>
            <<<numBlocks, BLOCK_SIZE>>>(g_ikeys, g_globalHistogram, n)));
}

template<typename T,
        const uint32_t RADIX,
        const uint32_t RADIX_LOG 
        >
inline void countExclusiveScan(uint32_t *g_globalHistogram, uint32_t *g_exclusiveCount) {
    const uint32_t numBlocks = sizeof(T) * 8 / RADIX_LOG;
    checkCUDA((countExclusiveScanImpl<T><<<numBlocks, RADIX / 2, RADIX>>>(g_globalHistogram, g_exclusiveCount)));
}

template<typename T, 
        typename TIndex,
        const uint32_t ITEMS_PER_THREAD,
        const uint32_t BLOCK_SIZE,
        const uint32_t RADIX,
        const uint32_t RADIX_LOG,
        const uint32_t RADIX_MASK
        >
inline void globalBinning(T *g_ikeys, T *g_okeys, TIndex *g_ivalsIndices, TIndex *g_ovalsIndices, uint32_t *g_exclusiveCount, uint32_t *g_atomicTileAssignCounter, uint32_t *g_statusCounter, uint32_t n, uint32_t currentDigit) {
    const uint32_t numBlocks = (n + ITEMS_PER_THREAD * BLOCK_SIZE - 1) / (ITEMS_PER_THREAD * BLOCK_SIZE);
    checkCUDA((globalBinningImpl<T, TIndex, ITEMS_PER_THREAD, BLOCK_SIZE, RADIX, RADIX_LOG, RADIX_MASK>
            <<<numBlocks, BLOCK_SIZE>>>(g_ikeys, g_okeys, g_ivalsIndices, g_ovalsIndices, g_exclusiveCount, g_atomicTileAssignCounter, g_statusCounter, n, currentDigit)));
}

template<typename T,
        const uint32_t RADIX = DEFAULT_RADIX, 
        const uint32_t RADIX_LOG = DEFAULT_RADIX_LOG, 
        const uint32_t ITEMS_PER_THREAD = DEFAULT_BINNING_ITEMS_PER_THREAD, 
        const uint32_t BLOCK_SIZE = DEFAULT_BINNING_BLOCK_SIZE
        >
uint32_t getDeviceRadixSortTempMemSize(uint32_t n) {
    // size of 'g_globalHistogram' + size of 'g_exclusiveCount' + size of 'g_atomicTileAssignCounter' + size of 'g_statusCounter'
    return ((sizeof(T) << 8) >> RADIX_LOG) * RADIX * sizeof(uint32_t) + sizeof(uint32_t) * RADIX * ((sizeof(T) << 8) >> RADIX_LOG)
           + sizeof(uint32_t) + sizeof(uint32_t) * RADIX * (n + (ITEMS_PER_THREAD * BLOCK_SIZE) - 1)/(ITEMS_PER_THREAD * BLOCK_SIZE);
}


/**
* @brief Device Radix Sort
*
* @tparam T: type of keys (should be uint32_t or uint64_t)
* @tparam TIndex: type of indices (should be uint32_t or uint64_t)
* @tparam RADIX: radix of the sort (should be 256 by default)
* @tparam RADIX_LOG: log of the radix (should be 8 by default)
* @tparam RADIX_MASK: mask of the radix (should be 255 by default)
* @tparam ITEMS_PER_THREAD: number of items per thread durring binning (should be 16)
* @tparam BLOCK_SIZE: size of the block (should be 256 by default)
* @tparam HIST_PART_SIZE: size of the histogram part processed by each block (should be 16384 by default)
* @tparam HIST_SUB_BLOCKS: number of histogram sub-blocks (should be 4 by default)
* 
* @param tempMem: temporary memory
* @param tempMemSize: size of temporary memory
* @param g_ikeys: input keys
* @param g_okeys: output keys
* @param g_ivalsIndices: input indices
* @param g_ovalsIndices: output indices
* @param n: number of elements
*/
template<typename T, typename TIndex,
        const uint32_t RADIX = DEFAULT_RADIX,
        const uint32_t RADIX_LOG = DEFAULT_RADIX_LOG,
        const uint32_t RADIX_MASK = RADIX - 1,
        const uint32_t ITEMS_PER_THREAD = DEFAULT_BINNING_ITEMS_PER_THREAD,
        const uint32_t BLOCK_SIZE = DEFAULT_BINNING_BLOCK_SIZE,
        const uint32_t HIST_PART_SIZE = DEFAULT_HIST_PART_SIZE,
        const uint32_t HIST_SUB_BLOCKS = DEFAULT_HIST_SUB_BLOCKS
        >
void deviceRadixSort(uint32_t* tempMem, uint32_t tempMemSize, T *g_ikeys, T *g_okeys, TIndex *g_ivalsIndices, TIndex *g_ovalsIndices, uint32_t n) {
    // compute the global histogram
    assert(tempMemSize >= getDeviceRadixSortTempMemSize<T>(n));

    // clean temporary memory
    checkCUDA((cudaMemset(tempMem, 0, tempMemSize)));

    // front scan kernel
    globalHistogram<T, HIST_PART_SIZE, HIST_SUB_BLOCKS, BLOCK_SIZE, RADIX, RADIX_LOG, RADIX_MASK>(g_ikeys, tempMem, n);

    // Print all values in global exclusive count
    uint32_t* h_hist = new uint32_t[RADIX];
    checkCUDA((cudaMemcpy(h_hist,
                          tempMem,
                          sizeof(uint32_t) * RADIX,
                          cudaMemcpyDeviceToHost)));

    // count exclusive scan
    countExclusiveScan<T, RADIX, RADIX_LOG>(tempMem, tempMem + RADIX * ((sizeof(T) << 8) >> RADIX_LOG));

    // binning kernel, run for each digit
    for (uint32_t digit = 0; digit < ((sizeof(T) << 8) >> RADIX_LOG); digit++) {

        // reset the counters
        checkCUDA((cudaMemset(
                tempMem + RADIX * ((sizeof(T) << 8) >> RADIX_LOG) * 2,
                0,
                sizeof(uint32_t) * RADIX * (n + (ITEMS_PER_THREAD * BLOCK_SIZE) - 1)/(ITEMS_PER_THREAD * BLOCK_SIZE) + sizeof(uint32_t)
        )));

        globalBinning<T, TIndex, ITEMS_PER_THREAD, BLOCK_SIZE, RADIX, RADIX_LOG, RADIX_MASK>(
            digit %2 == 0 ? g_ikeys : g_okeys,
            digit %2 == 1 ? g_ikeys : g_okeys,
            digit %2 == 0 ? g_ivalsIndices : g_ovalsIndices,
            digit %2 == 1 ? g_ivalsIndices : g_ovalsIndices,
            /* exclusive count */
            tempMem + RADIX * ((sizeof(T) << 8) >> RADIX_LOG),  
            /* atomic tile assign counter (put at the end of memory block to avoid unaligned access)*/
            tempMem + RADIX * ((sizeof(T) << 8) >> RADIX_LOG) * 2 + sizeof(uint32_t) * RADIX * (n + (ITEMS_PER_THREAD * BLOCK_SIZE) - 1)/(ITEMS_PER_THREAD * BLOCK_SIZE),
            /* status counter */
            tempMem + RADIX * ((sizeof(T) << 8) >> RADIX_LOG) * 2,  
            n, 
            digit);

        if (digit %2 == 1 && digit == ((sizeof(T) << 8) >> RADIX_LOG) - 1){
            // copy g_ikeys to g_okeys
            checkCUDA((cudaMemcpy(g_okeys, g_ikeys, sizeof(T) * n, cudaMemcpyDeviceToDevice)));
        }
    }
}


#endif //CUDATESTS_DEVICESORT_CUH
