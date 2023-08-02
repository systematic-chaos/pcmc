#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <mpi.h>
#include "ctimer.h"

float messageRoundtrip(char *msg);

#define N 4096
#define PING 0
#define PONG 1

static const char TEXT[] = "Así atravesamos la Mancha..., solitario país donde el sol está en su reino, y el hombre parece obra exclusiva del sol y del polvo; país entre todos famoso desde que el mundo entero se ha acostumbrado a suponer la inmensidad de sus llanuras recorrida por el caballo de D. Quijote. (...) Esto es lo cierto: la Mancha, si alguna belleza tiene, es la belleza de su conjunto, es su propia desnudez y monotonía, que si no distraen ni suspenden la imaginación, la dejan libre, dándole espacio y luz donde se precipite sin tropiezo alguno. La grandeza del pensamiento de don Quijote, no se comprende sino en la grandeza de la Mancha. En un país montuoso, fresco, verde, poblado de agradables sombras, con lindas casas, huertos floridos, luz templada y ambiente espeso, D. Quijote no hubiera podido existir, y habría muerto en flor, tras la primera salida, sin asombrar al mundo con las grandes hazañas de la segunda. D. Quijote necesitaba aquel horizonte, aquel suelo sin caminos, y que, sin embargo, todo él es camino; aquella tierra sin direcciones, pues por ella se va a todas partes, sin ir determinadamente a ninguna; tierra surcada por las veredas del acaso, de la aventura, y donde todo cuanto pase ha de parecer obra de la casualidad o de los genios de la fábula; necesitaba de aquel sol que derrite los sesos y hace locos a los cuerdos, aquel campo sin fin, donde se levanta el polvo de imaginarias batallas, produciendo al transparentar de la luz, visiones de ejércitos de gigantes, de torres, de castillos; necesitaba aquella escasez de ciudades, que hace más rara y extraordinaria la presencia de un hombre, o de un animal; necesitaba aquel silencio cuando hay calma, y aquel desaforado rugir de los vientos cuando hay tempestad; calma y ruido que son igualmente tristes y extienden su tristeza a todo lo que pasa, de modo que si se encuentra un ser humano en aquellas soledades, al punto se le tiene por un desgraciado, un afligido, un menesteroso, un agraviado que anda buscando quien lo ampare contra los opresores y tiranos; necesitaba, repito, aquella total ausencia de obras humanas que representen el positivismo, el sentido práctico, cortapisas de la imaginación, que la detendrían en su insensato vuelo; necesitaba, en fin, que el hombre no pusiera en aquellos campos más muestras de su industria y de su ciencia que los patriarcales molinos de viento, los cuales no necesitaban sino hablar, para asemejarse a colosos inquietos y furibundos, que desde lejos llaman y espantan al viajero con sus gestos amenazadores.";

/**
 * MPI program that computes the machine's values for `T` and `B`.
 * A ping-pong algorithm must be implemented for it, which sends a message from
 * a processor to another one. Next, the second processor sends back the message
 * to the first, and the roundtrip time is measured.
 * If the message size is 0 bytes (or a small number), time will basically denote
 * the `B` term. If the message size is large enough, time will basically denote
 * the `T` term.
 * Don't forget to repeat the sending a high number of times to avoid errors
 * in the measurement of small times.
 */
int main(int argc, char **argv) {
    float T, B;
    int myRank;
    char smallMessage, *largeMessage;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Data generation
    smallMessage = '\0';
    largeMessage = malloc(N * sizeof(char));
    strcpy(largeMessage, TEXT);

    // Small message roundtrip
    B = messageRoundtrip(&smallMessage);

    // Large message roundtrip
    T = messageRoundtrip(largeMessage);

    // Display results
    if (!myRank) {
        printf("B =\t%.0f ns\n", B);
        printf("T =\t%.0f ns\n", T);
    }

    MPI_Finalize();
    return 0;
}

float messageRoundtrip(char *msg) {
    int myRank, length, tag, i;
    double t1, t2, tucpu, tscpu;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    length = strlen(msg) + 1;
    tag = getppid();

    ctimer(&t1, &tucpu, &tscpu);
    switch (myRank) {
        case PING:
            for (i = 0; i < N; i++) {
                MPI_Send(msg, length, MPI_CHAR, PONG, tag, MPI_COMM_WORLD);
                MPI_Recv(msg, length, MPI_CHAR, PONG, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        case PONG:
            for (i = 0; i < N; i++) {
                MPI_Recv(msg, length, MPI_CHAR, PING, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(msg, length, MPI_CHAR, PING, tag, MPI_COMM_WORLD);
            }
            break;
    }
    ctimer(&t2, &tucpu, &tscpu);

    return (float)(t2 - t1) / N * 1e9;
}
