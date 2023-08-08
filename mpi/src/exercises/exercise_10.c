#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include "ctimer.h"

#define N 256
#define MIN(a, b) ( a < b ? a : b )

void matrixInitialization(int *adjacency, int *minCost, unsigned int dim);
void floyd(int *minCost, int *nextStep, int srcVtx, unsigned int dim);

/**
 * MPI program that computes the matrix of minimum distances in a directed graph
 * using Floyd's algorithm.
 */
int main(int argc, char **argv) {
    int *adjacencyMatrix, *minCostMatrix, *nextStepMatrix;
    int myRank, numProcessors, blockSize;
    int i, j, xPos, yPos, startVtx;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    blockSize = N / numProcessors;

    // Data generation
    adjacencyMatrix = malloc(N * N * sizeof(int));
    minCostMatrix = malloc(N * N * sizeof(int));
    nextStepMatrix = malloc((myRank ? blockSize : N) * N * sizeof(int));
    if (!myRank) {
        srand(time(NULL));
        xPos = rand() % N;
        yPos = rand() % N;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                adjacencyMatrix[i * N + j] = rand() % 2 ? rand() % (N * N) : -1;
            }
        }
        matrixInitialization(adjacencyMatrix, minCostMatrix, N);
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            floyd(minCostMatrix, nextStepMatrix + i * N, i, N);
        }
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential Floyd algorithm:\tF[%d][%d] = %d\n", xPos, yPos, minCostMatrix[yPos * N + xPos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast to other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Bcast(adjacencyMatrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    matrixInitialization(adjacencyMatrix, minCostMatrix, N);
    startVtx = myRank * blockSize;
    for (i = 0; i < blockSize; i++) {
        floyd(minCostMatrix, nextStepMatrix + i * N, startVtx + i, N);
    }

    // Result gathering in the root processor
    MPI_Gather(minCostMatrix + startVtx * N, blockSize * N, MPI_INT,
        minCostMatrix, blockSize * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(nextStepMatrix, blockSize * N, MPI_INT,
        nextStepMatrix, blockSize * N, MPI_INT, 0, MPI_COMM_WORLD);
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel Floyd algorithm:\tF[%d][%d] = %d\n", xPos, yPos, minCostMatrix[yPos * N + xPos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}

void matrixInitialization(int *adjacency, int *minCost, unsigned int dim) {
    unsigned int i, j, pos;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            pos = i * dim + j;
            if (i == j) {
                minCost[pos] = 0;
            } else if (adjacency[pos] > 0) {
                minCost[pos] = adjacency[pos];
            } else {
                minCost[pos] = USHRT_MAX;
            }
        }
    }
}

void floyd(int *minCost, int *nextStep, int srcVtx, unsigned int dim) {
    unsigned int j, k;
    int altDist, *posPtr;
    for (k = 0; k < dim; k++) {
        for (j = 0; j < dim; j++) {
            posPtr = minCost + srcVtx * dim + j;
            altDist = minCost[srcVtx * dim + k] + minCost[j * dim + k];
            if (*posPtr > altDist) {
                *posPtr = altDist;
                nextStep[j] = k + 1;
            }
        }
    }
}
