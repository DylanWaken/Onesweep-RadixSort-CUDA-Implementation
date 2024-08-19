//
// Created by dylan on 13/08/24.
//

/**
 * @file EWAMacros.cuh, the general purposed macros for cuda error checking, constants, etc.
 */

#ifndef EWAMACROS_CUH
#define EWAMACROS_CUH

// cuda includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARP_INDEX
#define WARP_INDEX (threadIdx.x / WARP_SIZE)
#endif

#ifndef LANE_INDEX
#define LANE_INDEX (threadIdx.x % WARP_SIZE)
#endif


#ifndef checkCUDNN
#define checkCUDNN(expression){ \
    cudaDeviceSynchronize();                            \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      logFatal(io::LOG_SEG_COMP, "Cudnn failed, error : ");  \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      assert(false);                                         \
    }                                                        \
}
#endif

#ifndef checkCUBLAS
#define checkCUBLAS(expression){ \
    cudaDeviceSynchronize();                            \
    cublasStatus_t status = (expression); \
    cudaDeviceSynchronize();                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      logFatal(io::LOG_SEG_COMP, "Cublas failed, error : ");  \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cublasGetStatusString(status) << std::endl; \
      assert(false);                                         \
    }                                                        \
}
#endif

#ifndef checkCUDA
#define checkCUDA(expression) { \
    expression; \
    cudaDeviceSynchronize(); \
    cudaError_t status = cudaGetLastError(); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at: " __FILE__ << ":" << __LINE__ << " <> "<< cudaGetErrorString(status) << std::endl; \
        assert(false); \
    } \
}
#endif

#endif // EWAMACROS_CUH

