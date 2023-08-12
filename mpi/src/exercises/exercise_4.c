#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "ctimer.h"

#define N 16384000

/**
 * MPI program that computes the scalar product of two `n`-length double vectors,
 * spanning a maximum of `p` processes. Process 0 must generate data, distribute
 * them to other processes, and retrieve and display final results.
 */
int main(int argc, char **argv) {
    double *a, *b, *x, *aAux, *bAux, *xAux;
    int myRank, numProcessors, chunkSize;
    unsigned int i, pos;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    chunkSize = N / numProcessors;

    // Data generation
    xAux = malloc(chunkSize * sizeof(double));
    aAux = malloc(chunkSize * sizeof(double));
    bAux = malloc(chunkSize * sizeof(double));
    if (!myRank) {
        a = malloc(N * sizeof(double));
        b = malloc(N * sizeof(double));
        x = malloc(N * sizeof(double));
        srand(time(NULL));
        for (i = 0; i < N; i++) {
            a[i] = rand() % (chunkSize / numProcessors);
            b[i] = rand() % (chunkSize / numProcessors);
        }
        pos = rand() % N;
    }

    // Sequential problem solving by the root process
    if (!myRank) {
        ctimer(&t1, &tucpu, &tscpu);
        for (i = 0; i < N; i++) {
            x[i] = a[i] * b[i];
        }
        ctimer(&t2, &tucpu, &tscpu);
        // Sequential results display
        printf("Sequential vector product x[%d] =\t%.0f\n", pos, x[pos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    // Data broadcast to other processes
    ctimer(&t1, &tucpu, &tscpu);
    MPI_Scatter(a, chunkSize, MPI_DOUBLE, aAux, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, chunkSize, MPI_DOUBLE, bAux, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    for (i = 0; i < chunkSize; i++) {
        xAux[i] = aAux[i] * bAux[i];
    }

    // Result gathering in the root process
    MPI_Gather(xAux, chunkSize, MPI_DOUBLE, x, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ctimer(&t2, &tucpu, &tscpu);

    // Results display
    if (!myRank) {
        printf("Parallel vector product x[%d] =\t%.0f\n", pos, x[pos]);
        printf("Time:\t%.9f seconds\n", (float)(t2 - t1));
    }

    MPI_Finalize();
    return 0;
}
