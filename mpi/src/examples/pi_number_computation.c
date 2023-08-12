#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define N 10000000000

double f(double x);

int main(int argc, char **argv) {
    double x, h, piSum, totalPiSeq, totalPiMpi;
    int taskId, totalTasks;
    unsigned long int i;
    double t1, t2, tucpu, tscpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalTasks);
    h = 1. / (double)N;

    // Sequential computation
    if (!taskId) {
        ctimer(&t1, &tucpu, &tscpu);
        piSum = 0.;
        for (i = 0; i < N; i++) {
            x = (i + 0.5) * h;
            piSum += f(x);
        }
        totalPiSeq = piSum * h;
        ctimer(&t2, &tucpu, &tscpu);
        printf("Sequential algorithm:\n");
        printf("pi is approximately %.16f (%ld intervals)\n", totalPiSeq, N);
        printf("Time:\t%f seconds\n", (float)(t2 - t1));
    }

    // Parallel computation using MPI
    ctimer(&t1, &tucpu, &tscpu);
    piSum = 0.;
    for (i = taskId; i < N; i += totalTasks) {
        x = (i + 0.5) * h;
        piSum += f(x);
    }
    MPI_Reduce(&piSum, &totalPiMpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    ctimer(&t2, &tucpu, &tscpu);

    if (!taskId) {
        totalPiMpi *= h;

        printf("\nAlgorithm parallelized using MPI:\n");
        printf("pi is approximately %.16f (%ld intervals)\n", totalPiMpi, N);
        printf("Time:\t%f seconds\n", (float)(t2 - t1));
        printf("Result difference Sequential-Parallel: %.16f\n", fabs(totalPiSeq - totalPiMpi));
    }

    return 0;
}

double f(double x) {
    return 4. / (1. + x * x);
}
