/*
 * CUDA program that computes the product of two matrices of `m * n`
 * unsigned long integer elements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "ctimer.h"

typedef unsigned long long int ullong;

#define RAND_INT() (rand() / (RAND_MAX / 10))

#define M 2048
#define N 512
#define CHUNK_SIZE 8

// CUDA Kernel code for the Device
__global__ void ex5MatMult(const long* A, const long* B, ullong* C) {

    /* These variables accumulate the partial sums for this output cell.
       Variable `accCellSum` is shared for all the threads in this block,
       that make for the same output cell. Since operations on this variable are
       performed on an atomic (mutual exclusion) way, this avoiding race conditions,
       the thread computes its partial accumulates in a local variable, and the
       shared variable is written (accessed) just once per thread. */
    __shared__ ullong accCellSum;
    ullong cell = accCellSum = 0;
    __syncthreads();

    unsigned int aIndex, bIndex;

    /* A single thread can locally compute a sequence of partial accumulates,
       decreasing the number of threads in the same block and the number
       of atomic operations on the same shared variable (hence, minimizing
       wait periods due to the serialization of concurrent accesses to the
       same resource). This must be balanced with an enough number of threads
       per block, finding a balance between parallelization and synchronization. */
    if (CHUNK_SIZE > 1) {
        for (int n = 0; n < CHUNK_SIZE; n++) {
            aIndex = blockIdx.y * N + threadIdx.x * CHUNK_SIZE + n;
            bIndex = blockIdx.x + (threadIdx.x * CHUNK_SIZE + n) * M;
            cell += A[aIndex] * B[bIndex];
        }
    } else {
        aIndex = blockIdx.y * N + threadIdx.x;
        bIndex = blockIdx.x + threadIdx.x * M;
        cell = A[aIndex] * B[bIndex];
    }
    atomicAdd(&accCellSum, cell);
    
    /* Upon the complete addition of partial accumulates, write the result back
       to the global memory address matching the output cell. */
    __syncthreads();
    if (!threadIdx.x) {
        C[blockIdx.y * gridDim.x + blockIdx.x] = accCellSum;
    }
}

unsigned long checkMatMult(const long* A, const long* B, const ullong* C) {
    unsigned long accError = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            ullong serialCell = 0;
            for (int k = 0; k < N; k++) {
                serialCell += A[i * N + k] * B[j + k * M];
            }

            accError += labs(C[i * M + j] - serialCell);
        }
    }

    return accError;
}

// Main program in the Host
int main(void) {
    double t1, t2, tucpu, tscpu;
    
    // Allocate memory in the host for the input and output matrices
    const size_t inputMatrixMemorySize = M * N * sizeof(long);
    const size_t outputMatrixMemorySize = M * M * sizeof(ullong);
    long *h_A = (long*)malloc(inputMatrixMemorySize);
    long *h_B = (long*)malloc(inputMatrixMemorySize);
    ullong *h_C = (ullong*)malloc(outputMatrixMemorySize);

    // Initialize the input matrices in the Host
    srand(time(NULL));
    for (unsigned int ij = 0; ij < M * N; ij++) {
        h_A[ij] = RAND_INT();
        h_B[ij] = RAND_INT();
    }

    // Allocate memory in the Device for the input and output matrices
    long *d_A = NULL; long *d_B = NULL; ullong *d_C = NULL;
    cudaMalloc(&d_A, inputMatrixMemorySize);
    cudaMalloc(&d_B, inputMatrixMemorySize);
    cudaMalloc(&d_C, outputMatrixMemorySize);

    // Copy the input matrices A & B from the Host's memory to the Device's
    cudaMemcpy(d_A, h_A, inputMatrixMemorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, inputMatrixMemorySize, cudaMemcpyHostToDevice);

    /* Allocate a grid as large as the (bidimensional) output matrix,
       and a number of threads per block that equals the number of additions to perform
       per output cell (this is, dimension N). */
    const dim3 blocksPerGrid(M, M);
    const int threadsPerBlock = N / CHUNK_SIZE;
    printf("Multiplying two matrices with dimensions %dx%d\n", M, N);
    printf("Launching the CUDA Kernel with a %dx%d grid and %d threads per block.\n",
        blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock);
    // Launch the CUDA Kernel `ex5MatMult`
    ctimer(&t1, &tucpu, &tscpu);
    ex5MatMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    ctimer(&t2, &tucpu, &tscpu);
    printf("GPU time:\t%.9f seconds\n", (float)(t2 - t1));

    // Copy the result output matrix from the Device's memory to the Host's
    cudaMemcpy(h_C, d_C, outputMatrixMemorySize, cudaMemcpyDeviceToHost);

    // Free memory in the Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display result data
    ctimer(&t1, &tucpu, &tscpu);
    unsigned long error = checkMatMult(h_A, h_B, h_C);
    ctimer(&t2, &tucpu, &tscpu);
    printf("Cumulative error: %lu\n", error);
    printf("CPU time:\t%.9f seconds\n", (float)(t2 - t1));

    // Free memory in the Host
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the Device and exit
    cudaDeviceReset();
    return 0;
}
