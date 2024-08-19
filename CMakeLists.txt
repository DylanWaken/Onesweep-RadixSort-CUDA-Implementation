cmake_minimum_required(VERSION 3.22)
project(CudaTests CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CudaTests main.cu DeviceScan.cuh DeviceSort.cuh cudaMacros.cuh)

# add cuda compiler flag: --generate-line-info
target_compile_options(CudaTests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --relocatable-device-code=true
        >)

set_target_properties(CudaTests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
        )
