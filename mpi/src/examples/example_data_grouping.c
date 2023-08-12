#include <stdio.h>
#include <mpi.h>

void sendData(int myRank, float *aPtr, float *bPtr, int *nPtr);

int main(int argc, char **argv) {
    int myRank, n;
    float a, b;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    sendData(myRank, &a, &b, &n);

    MPI_Finalize();
    return 0;
}

void sendData(int myRank, float *aPtr, float *bPtr, int *nPtr) {
    const int root = 0;       // Argument to MPI_Bcast
    char buffer[100];   // Argument to MPI_Pack/Unpack
    int position;       // and MPI_Bcast

    if (myRank == root) {
        printf("Enter a, b, and n\n");
        scanf("%f %f %d", aPtr, bPtr, nPtr);

        // Now pack the data into buffer
        position = 0;   // Start at the beginning of buffer
        MPI_Pack(aPtr, 1, MPI_FLOAT, buffer + position, 100, &position, MPI_COMM_WORLD);
        // Position has been incremented by sizeof(float) bytes
        MPI_Pack(bPtr, 1, MPI_FLOAT, buffer, 100, &position, MPI_COMM_WORLD);
        MPI_Pack(nPtr, 1, MPI_INT, buffer, 100, &position, MPI_COMM_WORLD);

        // Now broadcast contents of buffer
        MPI_Bcast(buffer, sizeof(float) + sizeof(float) + sizeof(int), MPI_PACKED, root, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(buffer, sizeof(float) + sizeof(float) + sizeof(int), MPI_PACKED, root, MPI_COMM_WORLD);
        
        // Now unpack the contents of buffer
        position = 0;
        MPI_Unpack(buffer, 100, &position, aPtr, 1, MPI_FLOAT, MPI_COMM_WORLD);
        // Once again position has been incremented by sizeof(float) bytes
        MPI_Unpack(buffer, 100, &position, bPtr, 1, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Unpack(buffer, 100, &position, nPtr, 1, MPI_INT, MPI_COMM_WORLD);

        printf("%f %f %d\n", *aPtr, *bPtr, *nPtr);
    }
}
