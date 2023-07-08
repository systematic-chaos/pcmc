# Parallel Computing Methods and Concepts

### Examples & Exercises

---

## MPI

1. MPI program that executes on four processes, sends a greetings message to the others and prints the greetings message from each process identifying its range.

2. MPI program that computes the sum of two vectors of `n` real elements expressed in simple precision, one of them multiplied by a constant `alpha` (operation saxpy).

3. MPI program that computes the sum of the `n` elements of a double vector, with and without using reduction instructions. The root process must generate data, distribute them to other processes, gather partial results and aggregate them into final results to be displayed.
Suggestion: Take the number of processors `p` as a power of 2 and `n` as a multiple of `p`.

4. MPI program that computes the scalar product of two `n`-length double vectors. Process 0 must generate data, distribute them to other processes, and retrieve and display final results.

5. MPI program that computes the machine network's values for `T` and `B`. A ping-pong algorithm must be implemented for it, which sends a message from a processor to another one. Next, the second processor sends back the message to the first, and the roundtrip time is measured.
If the message size is 0 bytes (or a small number), time will basically denote the `B` term. If the message size is large enough, time will basically denote the `T` term. Don't forget to repeat the sending a high number of times to avoid errors in the measurement of small times.

6. MPI program that cmputes the machine network's values for `T` and `B`, calculating a regression by least squares. A ping-pong algorithm must be implemented for it, which sends a message from a processor to another one. Next, the second processor sends back the message to the first. The roundtrip time will be measured for several message sizes. Empirical data must be adjusted to a line. The ordinate in the origin point will basically match the `b` term. The line's slope will basically match the `t` term.

7. MPI program that computes the euclidean norm of a float vector containing `n` elements. Suggestion: It must not fail in the euclidean norm is smaller than the maximum positive number that can be represented using _floating_ simple precision. The root process must generate data, distribute them to other processes, and retrieve, gather and display the final results.

8. MPI program that computes the integral, between `a` and `b`, of a polynomical function of grade `m`, splitting the interval `[a, b]` in `n` subintervals.

9. MPI program that multiplies two long integers stored as a couple of vectors.

10. MPI program that computes the matrix of minimum distances in a directed graph using Floyd's algorithm.
