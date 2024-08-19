//
// Created by dylan on 13/08/24.
//

#ifndef CUDATESTS_DEVICESCAN_CUH
#define CUDATESTS_DEVICESCAN_CUH

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> (LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cudaMacros.cuh"

template <typename T>
__global__ void sumImpl(size_t n, T* g_idata, T* g_odata, bool inclusive = true, bool collect_last = false,
                 T* g_last_element_array = nullptr){
    extern __shared__ T temp[];// allocated on invocation

    uint32_t thid = threadIdx.x;
    uint32_t offset = 1;
    uint32_t operative_size = 2 * blockDim.x;

    n = (blockIdx.x != gridDim.x - 1 || n % operative_size == 0) ? operative_size : n % operative_size;

    g_idata += blockIdx.x * operative_size;
    g_odata += blockIdx.x * operative_size;
    T last_input;

    // fetch data
    uint32_t ai = thid;
    uint32_t bi = thid + (operative_size/2);
    uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = ai < n ? g_idata[ai] : 0;           // load input into shared memory without bank conflicts
    temp[bi + bankOffsetB] = bi < n ? g_idata[bi] : 0;           // load input into shared memory without bank conflicts
    __syncthreads();
    if (thid == 0) last_input = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];

    // build sum in place up the tree
    #pragma unroll
    for (uint32_t d = operative_size>>1; d > 0; d >>= 1){
        __syncthreads();
        if (thid < d){
            uint32_t aip = offset*(2*thid+1)-1;
            uint32_t bip = offset*(2*thid+2)-1;
            aip += CONFLICT_FREE_OFFSET(aip);
            bip += CONFLICT_FREE_OFFSET(bip);
            temp[bip] += temp[aip];
        }
        offset *= 2;
    }

    if (thid==0) {
        temp[(operative_size-1) + CONFLICT_FREE_OFFSET(operative_size - 1)] = 0;
    } // clear the last element

    // traverse down tree & build scan
    #pragma unroll
    for (uint32_t d = 1; d < operative_size; d *= 2){
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            uint32_t aip = offset*(2*thid+1)-1;
            uint32_t bip = offset*(2*thid+2)-1;
            aip += CONFLICT_FREE_OFFSET(aip);
            bip += CONFLICT_FREE_OFFSET(bip);

            T t = temp[aip];
            temp[aip] = temp[bip];
            temp[bip] += t;
        }
    }

    __syncthreads();

    if (thid == 0) {
        T last = last_input + temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        if (inclusive) g_odata[n - 1] = last;
        if (collect_last) g_last_element_array[blockIdx.x] = last;
    }

    // write results to global memory
    if (inclusive) {
        if (ai > 0 && ai < n) {
            g_odata[ai - 1] = temp[ai + bankOffsetA];
        }
        if (bi > 0 && bi < n) g_odata[bi - 1] = temp[bi + bankOffsetB];
    } else {
        if (ai < n) g_odata[ai] = temp[ai + bankOffsetA];
        if (bi < n) g_odata[bi] = temp[bi + bankOffsetB];
    }
}

template <typename T>
__global__ void scanDistrib(T* g_sum_output, size_t n, uint32_t op_size, T* g_last_element_array) {
    uint32_t gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= n || gid < op_size) return;
    uint32_t pick_up = gid / op_size - 1;

    T last_element = g_last_element_array[pick_up];
    g_sum_output[gid] += last_element;
}

__forceinline__ size_t compute_temp_size(size_t n, uint32_t op_size){
    size_t temp_size = 0;
    while ((n + op_size - 1) / op_size > 1){
        uint32_t nNext = (n + op_size - 1) / op_size;
        temp_size += nNext;
        n = nNext;
    }
    return temp_size * 2;
}

/**
 * @brief an recursive inclusive index sum, optimized
 * @tparam T the type of the data
 * @tparam BLOCK_SIZE the block size
 * @param g_temp the temporary memory, used when n > 2 * BLOCK_SIZE and recursive call is needed
 * @param n the size of the input
 * @param g_idata the input data
 * @param g_odata the output data
 */
template <typename T, const uint32_t BLOCK_SIZE = 512>
void inclusiveSum(T* g_temp, size_t temp_size, T* g_idata, T* g_odata,  size_t n){
    dim3 block(BLOCK_SIZE);
    uint32_t op_size = 2 * BLOCK_SIZE;
    dim3 grid((n + op_size - 1) / op_size);
    assert(temp_size >= compute_temp_size(n, op_size));

    bool proceed_recursion = (n + op_size - 1) / op_size > 1;

    // run the sum on each block
    checkCUDA((sumImpl<T><<<grid, block, sizeof(T) * op_size
        + CONFLICT_FREE_OFFSET(op_size) * sizeof(T)>>>(n, g_idata, g_odata, true,proceed_recursion, g_temp)));

    if (proceed_recursion){
        uint32_t nNext = (n + op_size - 1) / op_size;

        // run the sum
        inclusiveSum<T, BLOCK_SIZE>(g_temp + 2 * nNext, temp_size - 2 * nNext, g_temp, g_temp + nNext, nNext);

        // distribute the last element
        checkCUDA((scanDistrib<T><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>
            (g_odata, n, op_size, g_temp + nNext)));
    }
}


template <typename T, const uint32_t BLOCK_SIZE = 512>
void exclusiveSum(T* g_temp, size_t temp_size, T* g_idata, T* g_odata,  size_t n){
    dim3 block(BLOCK_SIZE);
    uint32_t op_size = 2 * BLOCK_SIZE;
    dim3 grid((n + op_size - 1) / op_size);
    assert(temp_size >= compute_temp_size(n, op_size));

    bool proceed_recursion = (n + op_size - 1) / op_size > 1;

    // run the sum on each block
    checkCUDA((sumImpl<T><<<grid, block, sizeof(T) * op_size
          + CONFLICT_FREE_OFFSET(op_size) * sizeof(T)>>>(n, g_idata, g_odata, false, proceed_recursion, g_temp)));

    if (proceed_recursion){
        uint32_t nNext = (n + op_size - 1) / op_size;

        // run the sum (note we use inclusive sum for sub-problems)
        inclusiveSum<T, BLOCK_SIZE>(g_temp + 2 * nNext, temp_size - 2 * nNext, g_temp, g_temp + nNext, nNext);

        // distribute the last element
        checkCUDA((scanDistrib<T><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>
                (g_odata, n, op_size, g_temp + nNext)));
    }
}

#endif //CUDATESTS_DEVICESCAN_CUH
