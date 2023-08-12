#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "ctimer.h"

#define N 16384

/**
 * MPI program that computes the sum of two vectors of `n` real elements
 * expressed in simple precision, one of them multiplied by a constant
 * alpha (operation saxpy), using a maximum of `p` processes.
 */
int main(int argc, char **argv) {
    const float alpha = 3.5;
    float *a, *b, *aAux, *bAux, *xSeq, *xPar, *xAux;
    int myRank, numProcessors, chunkSize;
    unsigned int i, pos;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    chunkSize = N / numProcessors;

    // Data generation
    aAux = malloc(chunkSize * sizeof(float));
    bAux = malloc(chunkSize * sizeof(float));
    xAux = malloc(chunkSize * sizeof(float));
    if (!myRank) {
        a = malloc(N * sizeof(float));
        b = malloc(N * sizeof(float));
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            a[i] = rand() % N;
            b[i] = rand() % N;
        }
        pos = rand() % N;
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        xSeq = malloc(N * sizeof(float));
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            xSeq[i] = alpha * a[i] + b[i];
        }
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential saxpy x[%d] =\t%0.f\n", pos, xSeq[pos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast from root to the other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Scatter(a, chunkSize, MPI_FLOAT, aAux, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, chunkSize, MPI_FLOAT, bAux, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    for (i = 0; i < chunkSize; i++) {
        xAux[i] = alpha * aAux[i] + bAux[i];
    }

    // Result gathering in the root process
    if (!myRank) {
        xPar = malloc(N * sizeof(float));
    }
    MPI_Gather(xAux, chunkSize, MPI_FLOAT, xPar, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel saxpy x[%d] =\t%.0f\n", pos, xPar[pos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}
