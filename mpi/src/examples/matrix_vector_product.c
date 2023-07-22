#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define M 6400
#define N 3200

// MATRIX-VECTOR MULTIPLICATION    x = A * b
int main(int argc, char **argv) {
    int taskId, nTasks;
    double *A, *Ablock, *b, *z;
    double *xSeq, *xMpi;
    double sum;
    int i, j;
    double t1, t2, tucpu, tscpu;

    A = malloc(M * N * sizeof(double));
    b = malloc(M * sizeof(double));
    xSeq = malloc(M * sizeof(double));
    xMpi = malloc(M * sizeof(double));

    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
    MPI_Comm_size(MPI_COMM_WORLD, &nTasks);

    Ablock = malloc(M * N / nTasks * sizeof(double));
    z = malloc(M / nTasks * sizeof(double));

    // Data generation
    if (!taskId) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                *(A + i + j * M) = rand() % nTasks;
            }
            *(b + i) = rand() % nTasks;
        }
    }

    // Sequential matrix-vector multiplication
    if (!taskId) {
        ctimer(&t1, &tucpu, &tscpu);

        for (i = 0; i < M; i++) {
            sum = 0.;
            for (j = 0; j < N; j++) {
                sum += *(A + i * N + j) * *(b + j);
            }
            *(xSeq + i) = sum;
        }

        ctimer(&t2, &tucpu, &tscpu);
        printf("MatxVec sequential product x[5] = %f\n", xSeq[5]);
        printf("Time:\t%f seconds\n", (float)(t2 - t1));
    }

    // Parallel Matrix-Vector multiplication
    if (!taskId) {
        ctimer(&t1, &tucpu, &tscpu);
    }

    MPI_Bcast(b, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, M * N / nTasks, MPI_DOUBLE, Ablock, M * N / nTasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = 0; i < M / nTasks; i++) {
        sum = 0.;
        for (j = 0; j < N; j++) {
            sum += *(Ablock + i * N + j) * *(b + j);
        }
        *(z + i) = sum;
    }

    MPI_Gather(z, M / nTasks, MPI_DOUBLE, xMpi, M / nTasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Results output
    if (!taskId) {
        ctimer(&t2, &tucpu, &tscpu);

        printf("Parallel MatxVec product x[5] = %f\n", xMpi[5]);
        printf("Time:\t%f seconds\n", (float)(t2 - t1));

        printf("Checking...\n");
        sum = 0.;
        for (i = 0; i < M; i++) {
            sum += (xSeq[i] - xMpi[i]) * (xSeq[i] - xMpi[i]);
        }
        sum = sqrt(sum);
        printf("Differential norm SEQUENTIAL - PARALLEL = %f\n", sum);
    }

    MPI_Finalize();
    return 0;
}
