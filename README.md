# Header Only OneSweep Radix Sort

This is an implementation of the OneSweep Radix Sort algorithm in C++ with CUDA. 
The implementation is based on the following paper:

[Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784) by Andy Adinets, Duane Merrill

This implementation is deeply anotated, so feel free to learn and optimize this algorithm

## How to build

1: clone the repository

2: make a build directory
```
mkdir build
cd build
```
3: run cmake
```
cmake ..
```
4: build
```
make
```
The excutable should be in the build directory

## How to use in your project

copy the header files
```
cudaMacros.cuh
DeviceSort.cuh
DeviceScan.cuh
```
to your project
