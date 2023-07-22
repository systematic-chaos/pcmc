#include <stdio.h>
#include <mpi.h>

void sendData(int myRank, int *nPtr, double *aPtr, double *bPtr);

int main(int argc, char **argv) {
    int myRank, n;
    double a, b;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    sendData(myRank, &n, &a, &b);

    MPI_Finalize();
    return 0;
}

void sendData(int myRank, int *nPtr, double *aPtr, double *bPtr) {
    const int root = 0; // Arguments for MPI_Bcast
    int count;

    if (myRank == root) {   // If I am the root process
        printf("Input n, a & b\n");
        scanf("%d %lf %lf", nPtr, aPtr, bPtr);
    }

    // Everyone execute MPI_Bcast
    MPI_Bcast(nPtr, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(aPtr, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(bPtr, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
}
