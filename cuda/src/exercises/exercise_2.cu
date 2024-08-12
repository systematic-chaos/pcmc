/*
 * CUDA program that computes the sum of two matrices of `m * n` size.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define RAND_FLOAT() (rand() / (float)RAND_MAX)

#define M 256
#define N 256

// CUDA Kernel code for the Device
__global__ void ex2MatSum(const float* A, const float* B, float* C) {
    const unsigned int n = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (n < M * N) {
        C[n] = A[n] + B[n];
    }
}

float checkMatSum(const float* A, const float* B, const float* C) {
    float accError = 0;
    unsigned int index;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            index = i * N + j;
            accError += fabs(A[index] + B[index] - C[index]);
        }
    }

    return accError;
}

// Main program in the Host
int main(void) {
    
    // Allocate memory in the Host for the input and output matrices
    const size_t memorySize = M * N * sizeof(float);
    float *h_A = (float*)malloc(memorySize);
    float *h_B = (float*)malloc(memorySize);
    float *h_C = (float*)malloc(memorySize);

    // Initialize the input matrices in the Host
    const unsigned int numElements = M * N;
    for (unsigned int ij = 0; ij < numElements; ij++) {
        h_A[ij] = RAND_FLOAT();
        h_B[ij] = RAND_FLOAT();
    }

    // Allocate memory in the Device for the input and output matrices
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc(&d_A, memorySize);
    cudaMalloc(&d_B, memorySize);
    cudaMalloc(&d_C, memorySize);

    // Copy the input matrices A & B from the Host's memory to the Device's
    cudaMemcpy(d_A, h_A, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, memorySize, cudaMemcpyHostToDevice);

    /* The grid could be a single-dimensional array of blocks,
       as the input and output matrices are, but this a good chance
       to operate with a multi(two)-dimensional grid instead.
       This is handled as a demonstrative exercise, even if an unusually
       low rate of threads per block is allocated. */
    const dim3 blocksPerGrid(M, M);
    const int threadsPerBlock = N / M;
    printf("Multiplying two matrices with dimensions %dx%d\n", M, N);
    printf("Launching the CUDA Kernel with a %dx%d grid and %d threads per block.\n",
        blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock);
    // Launch the CUDA Kernel `ex2MatSum`
    ex2MatSum<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy the result output matrix from the Device's memory to the Host's
    cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    float error = checkMatSum(h_A, h_B, h_C);
    printf("Cumulative error: %.4f\n", error);

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();
    return 0;
}
