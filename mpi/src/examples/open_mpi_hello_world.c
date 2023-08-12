/* "Hello World!" MPI Test Program */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

int main(int argc, char** argv) {
    char buf[256];
    int myRank, numProcs;
    
    // Initialize the infrastructure necessary for communication
    MPI_Init(&argc, &argv);
    
    // Identify this process
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    // Find out how many total processes are active
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    /* Until this point, all programs have been doing exactly the same.
     * Here, we check the rank to distinguish the roles of the programs. */
    if (!myRank) {
        int otherRank;
        printf("We have %i processes.\n", numProcs);
        
        // Send messages to all other processes
        for (otherRank = 1; otherRank < numProcs; otherRank++) {
            sprintf(buf, "Hello %i!", otherRank);
            MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, otherRank, 0, MPI_COMM_WORLD);
        }
        
        // Receive messages from all other processes
        for (otherRank = 1; otherRank < numProcs; otherRank++) {
            MPI_Recv(buf, 256, MPI_CHAR, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", buf);
        }
    } else {
        // Receive message from process #0
        MPI_Recv(buf, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        assert(memcmp(buf, "Hello ", 6) == 0);
        
        // Send message to process #0
        sprintf(buf, "Process %i reporting for duty.", myRank);
        MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    // Tear down the communication infrastructure
    MPI_Finalize();
    return 0;
}

