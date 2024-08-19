#include <iostream>

#include "DeviceSort.cuh"

#define TEST_SIZE (1024 * 1024)

void checkScan() {
    // Allocate host memory
    const int size = 1000;
    uint32_t* h_input = new uint32_t[size];
    uint32_t* h_inclusive_output = new uint32_t[size];
    uint32_t* h_exclusive_output = new uint32_t[size];

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 10;  // Random numbers between 0 and 9
    }

    // Allocate device memory
    uint32_t* d_input;
    uint32_t* d_inclusive_output;
    uint32_t* d_exclusive_output;
    uint32_t* d_temp;
    cudaMalloc(&d_input, size * sizeof(uint32_t));
    cudaMalloc(&d_inclusive_output, size * sizeof(uint32_t));
    cudaMalloc(&d_exclusive_output, size * sizeof(uint32_t));
    uint32_t temp_size = compute_temp_size(size, 512);
    cudaMalloc(&d_temp, temp_size * sizeof(uint32_t));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Perform inclusive sum
    inclusiveSum(d_temp, temp_size, d_input, d_inclusive_output, size);

    // Perform exclusive sum
    exclusiveSum(d_temp, temp_size, d_input, d_exclusive_output, size);

    // Copy results back to host
    cudaMemcpy(h_inclusive_output, d_inclusive_output, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exclusive_output, d_exclusive_output, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Verify results
    uint32_t inclusive_sum = 0;
    uint32_t exclusive_sum = 0;
    bool inclusive_correct = true;
    bool exclusive_correct = true;

    for (int i = 0; i < size; i++) {
        inclusive_sum += h_input[i];
        if (h_inclusive_output[i] != inclusive_sum) {
            inclusive_correct = false;
            break;
        }

        if (h_exclusive_output[i] != exclusive_sum) {
            exclusive_correct = false;
            break;
        }
        exclusive_sum += h_input[i];
    }

    std::cout << "Inclusive sum test " << (inclusive_correct ? "passed" : "failed") << std::endl;
    std::cout << "Exclusive sum test " << (exclusive_correct ? "passed" : "failed") << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_inclusive_output;
    delete[] h_exclusive_output;
    cudaFree(d_input);
    cudaFree(d_inclusive_output);
    cudaFree(d_exclusive_output);
}

void testDeviceRadixSort(uint32_t test_size) {
    // Generate random uint32_t array
    uint32_t* h_input = new uint32_t[test_size];
    for (uint32_t i = 0; i < test_size; i++) {
        h_input[i] = rand() % (1<<30) + 1;
    }


    // Allocate device memory
    uint32_t* d_input;
    uint32_t* d_output;
    uint32_t* d_indices_in;
    uint32_t* d_indices_out;
    cudaMalloc(&d_input, sizeof(uint32_t) * test_size);
    cudaMalloc(&d_output, sizeof(uint32_t) * test_size);
    cudaMalloc(&d_indices_in, sizeof(uint32_t) * test_size);
    cudaMalloc(&d_indices_out, sizeof(uint32_t) * test_size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, sizeof(uint32_t) * test_size, cudaMemcpyHostToDevice);

    // Generate index array
    generateIndexArray<uint32_t>(d_indices_in, test_size);


    // Allocate temporary memory for radix sort
    uint32_t tempMemSize = getDeviceRadixSortTempMemSize<uint32_t>(test_size);
    uint32_t* d_tempMem;
    cudaMalloc(&d_tempMem, tempMemSize);

    // Perform device radix sort
    deviceRadixSort<uint32_t>(d_tempMem, tempMemSize, d_input, d_output, d_indices_in, d_indices_out, test_size);
    // Copy result back to host
    uint32_t* h_output = new uint32_t[test_size];
    cudaMemcpy(h_output, d_output, sizeof(uint32_t) * test_size, cudaMemcpyDeviceToHost);

    // Verify sorting
    bool sorted = true;
    for (uint32_t i = 1; i < test_size; i++) {
        if (h_output[i] < h_output[i - 1]) {
            sorted = false;
        }
    }

    std::cout << "Sorting test " << (sorted ? "passed" : "failed") << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices_in);
    cudaFree(d_indices_out);
    cudaFree(d_tempMem);
}



int main() {
    checkScan();
    // test by number of blocks and a non block offset
    testDeviceRadixSort(DEFAULT_BINNING_ITEMS_PER_THREAD * DEFAULT_BINNING_BLOCK_SIZE * 32 + 1031);
    return 0;
}
