#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // Variables initialization
    int myRank, processors, source, dest;
    char message[100];
    const int tag = 50;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);

    if (myRank) {
        sprintf(message, "Greetings from process %d", myRank);
        dest = 0;
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    } else {
        for (source = 1; source < processors; source++) {
            MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
    }

    MPI_Finalize();
    return 0;
}
