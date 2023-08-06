#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <math.h>
#include <mpi.h>
#include "ctimer.h"

#define MSG_MAX_SIZE 65536
#define SEND_TIMES 1024
#define PING 0
#define PONG 1

struct Line {
    float a;
    float b;
};

float messageRoundtrip(char *msg, unsigned int length);
void linearLeastSquares(float *x, float *y, int n, struct Line *line);

static const char TEXT[] = "Así atravesamos la Mancha..., solitario país donde el sol está en su reino, y el hombre parece obra exclusiva del sol y del polvo; país entre todos famoso desde que el mundo entero se ha acostumbrado a suponer la inmensidad de sus llanuras recorrida por el caballo de D. Quijote. (...) Esto es lo cierto: la Mancha, si alguna belleza tiene, es la belleza de su conjunto, es su propia desnudez y monotonía, que si no distraen ni suspenden la imaginación, la dejan libre, dándole espacio y luz donde se precipite sin tropiezo alguno. La grandeza del pensamiento de don Quijote, no se comprende sino en la grandeza de la Mancha. En un país montuoso, fresco, verde, poblado de agradables sombras, con lindas casas, huertos floridos, luz templada y ambiente espeso, D. Quijote no hubiera podido existir, y habría muerto en flor, tras la primera salida, sin asombrar al mundo con las grandes hazañas de la segunda. D. Quijote necesitaba aquel horizonte, aquel suelo sin caminos, y que, sin embargo, todo él es camino; aquella tierra sin direcciones, pues por ella se va a todas partes, sin ir determinadamente a ninguna; tierra surcada por las veredas del acaso, de la aventura, y donde todo cuanto pase ha de parecer obra de la casualidad o de los genios de la fábula; necesitaba de aquel sol que derrite los sesos y hace locos a los cuerdos, aquel campo sin fin, donde se levanta el polvo de imaginarias batallas, produciendo al transparentar de la luz, visiones de ejércitos de gigantes, de torres, de castillos; necesitaba aquella escasez de ciudades, que hace más rara y extraordinaria la presencia de un hombre, o de un animal; necesitaba aquel silencio cuando hay calma, y aquel desaforado rugir de los vientos cuando hay tempestad; calma y ruido que son igualmente tristes y extienden su tristeza a todo lo que pasa, de modo que si se encuentra un ser humano en aquellas soledades, al punto se le tiene por un desgraciado, un afligido, un menesteroso, un agraviado que anda buscando quien lo ampare contra los opresores y tiranos; necesitaba, repito, aquella total ausencia de obras humanas que representen el positivismo, el sentido práctico, cortapisas de la imaginación, que la detendrían en su insensato vuelo; necesitaba, en fin, que el hombre no pusiera en aquellos campos más muestras de su industria y de su ciencia que los patriarcales molinos de viento, los cuales no necesitaban sino hablar, para asemejarse a colosos inquietos y furibundos, que desde lejos llaman y espantan al viajero con sus gestos amenazadores.";

/**
 * MPI program that computes the machine's values for `T` and `B`, calculating
 * a regression by least squares.
 * A ping-pong algorithm must be implemented for it, which sends a message from
 * a processor to another one. Next, the second processor sends back the message
 * to the first. The roundtrip time will be measured for several message sizes.
 * Empirical data must be adjusted to a line. The ordinate in the origin point
 * will basically match the `b` term. The line's slope will basicall match the
 * `t` term.
 */
int main(int argc, char **argv) {
    float T, B;
    char *message;
    float *x, *y;
    struct Line equation;
    int myRank, n;
    unsigned int msgLength;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Data generation
    message = malloc(MSG_MAX_SIZE * sizeof(char));
    memset(message, 0, MSG_MAX_SIZE);
    strcpy(message, TEXT);
    x = malloc((log2(MSG_MAX_SIZE) + 1) * sizeof(int));
    y = malloc((log2(MSG_MAX_SIZE) + 1) * sizeof(int));
    
    // Send messages and measure delivery roundtrip time
    for (msgLength = 1, n = 0; msgLength <= MSG_MAX_SIZE; msgLength *= 2, n++) {
        x[n] = msgLength;
        y[n] = messageRoundtrip(message, msgLength);
    }

    if (!myRank) {
        // Approximate a function via the linear least squares method
        linearLeastSquares(x, y, n, &equation);

        // Extract the t and b factors from the function's formula and samples
        B = equation.a;
        T = equation.b;

        // Display results
        printf("B =\t%.5f\n", B);
        printf("T =\t%.5f\n", T);
        printf("y[0] =\t%.0f\n", y[0]);
    }

    MPI_Finalize();
    return 0;
}

float messageRoundtrip(char *msg, unsigned int length) {
    int myRank, tag, i;
    double t1, t2, tucpu, tscpu;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    tag = getppid();

    ctimer(&t1, &tucpu, &tscpu);
    switch (myRank) {
        case PING:
            for (i = 0; i < SEND_TIMES; i++) {
                MPI_Send(msg, length, MPI_CHAR, PONG, tag, MPI_COMM_WORLD);
                MPI_Recv(msg, length, MPI_CHAR, PONG, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            break;
        case PONG:
            for (i = 0; i < SEND_TIMES; i++) {
                MPI_Recv(msg, length, MPI_CHAR, PING, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(msg, length, MPI_CHAR, PING, tag, MPI_COMM_WORLD);
            }
            break;
    }
    ctimer(&t2, &tucpu, &tscpu);

    return (float)(t2 - t1) / SEND_TIMES * 1e9;
}

void linearLeastSquares(float *x, float *y, int n, struct Line *line) {
    float a, b, sumX = 0, sumX2 = 0, sumY = 0, sumXY = 0;
    int i;

    for (i = 0; i < n; i++) {
        sumX += x[i];
        sumX2 += x[i] * x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
    }

    b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    a = (sumY - b * sumX) / n;

    line -> a = a;
    line -> b = b;
}
