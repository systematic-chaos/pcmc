/*
 * Vector sum
 */

#include <stdio.h>
#include <stdlib.h>

// For understanding CUDA runtime routines (they start with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel code for the Device
 *
 * It computes the sum of two vectors A and B and stores it in C.
 * All vectors have the same number of elements: `numElements`.
 */

__global__ void vecSum(const float *A, const float *B, float *C, int numElements) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (n < numElements) {
        C[n] = A[n] + B[n];
    }
}

/*
 * Main program in the Host
 */
int main(void) {
    // Prints the vectors's length and computes their size
    const int numElements = 65536;
    size_t size = numElements * sizeof(float);
    printf("[Vector sum with %d elements]\n", numElements);

    // Allocate memory in the Host for the input vector A
    float *h_A = (float*)malloc(size);

    // Allocate memory in the Host for the input vector B
    float *h_B = (float*)malloc(size);

    // Allocate memory in the Host for the output vector C
    float *h_C = (float*)malloc(size);

    // Initialize the input vectors in the Host
    for (int n = 0; n < numElements; n++) {
        h_A[n] = rand() / (float)RAND_MAX;
        h_B[n] = rand() / (float)RAND_MAX;
    }

    // Allocate memory in the Device for the input vector A
    float *d_A = NULL;
    if (d_A) cudaFree(d_A);
    cudaMalloc((void**)&d_A, size);

    // Allocate memory in the Device for the input vector B
    float *d_B = NULL;
    if (d_B) cudaFree(d_B);
    cudaMalloc((void**)&d_B, size);

    // Allocate memory in the Device for the input vector C
    float *d_C = NULL;
    if (d_C) cudaFree(d_C);
    cudaMalloc((void**)&d_C, size);

    // Copy the input vectors A & B from the Host's memory to the Device's memory
    printf("Copy the input vectors A & B from the Host's memory to the Device's memory\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the CUDA Kernel `sumVec`
    const int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("The CUDA Kernel is launched with %d blocks of %d threads\n",
        blocksPerGrid, threadsPerBlock);
    vecSum<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy the result vector from the Device's memory to the Host's memory
    printf("Copy the output data from the Device's memory to the Host's memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    printf("Three first elements of A\t%f %f %f\n", h_A[0], h_A[1], h_A[2]);
    printf("Three first elements of B\t%f %f %f\n", h_B[0], h_B[1], h_B[2]);
    printf("Three first elements of C\t%f %f %f\n", h_C[0], h_C[1], h_C[2]);

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();

    return 0;
}
