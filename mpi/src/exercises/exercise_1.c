#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MSG_MAX_LEN 512

/**
 * MPI program that executes on four processes, sends a greetings message
 * to the others and prints the greetings message from each process
 * identifying its range.
 */
int main(int argc, char **argv) {
    char *buf, *msg;
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int worldSize, worldRank, msgLength, n;
    buf = malloc(MSG_MAX_LEN * sizeof(char));
    msg = malloc(MSG_MAX_LEN * sizeof(char));

    MPI_Init(&argc, &argv);

    // Data generation
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Get_processor_name(processorName, &msgLength);
    sprintf(msg, "Hello everyone from processor %s, ranked %d out of %d",
        processorName, worldRank + 1, worldSize);
    msgLength = strlen(msg) + 1;

    // Sequential problem solving by the root process
     if (!worldRank) {
        printf("%s\n", msg);
        strcpy(buf, msg);
     }

    // Data broadcast from root to the other processes
    MPI_Bcast(buf, msgLength, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Local computation in each processor
    for (n = 0; n < worldSize; n++) {
        if (n != worldRank) {
            MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, n, 0, MPI_COMM_WORLD);

            // Result gathering in the root process
            if (!worldRank) {
                MPI_Recv(buf, MSG_MAX_LEN, MPI_CHAR, n, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Results display
                printf("%s\n", buf);
            }
        }
    }


    // Results display

    MPI_Finalize();
    return 0;
}
