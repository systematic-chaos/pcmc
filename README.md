# Parallel Computing Methods and Concepts

### Examples & Exercises

---

## OpenMP

1. OpenMP program that executes on four threads, identifies the threads executing and the total number of threads, and displays results on screen.

2. OpenMP program that computes the sum of the `p` elements of a vector of doubles. The maximum number of threads to be expanded is `p / 2`.
Suggestion: Take `p` as power of two.

3. OpenMP program that computes the sum of the `n` elements of a vector of doubles, spanning a maximum of `p` threads, both with and without using reduction instructions.
Suggestion: Take `p` as power of 2 and `n` as a multiple of `p`.

4. OpenMP program that computes the element holding the greatest absolute value in a vector of doubles, using a maximum of `p` threads.

5. OpenMP program computing the scalar product of two double vectors of `n` elements each, spanning a maximum of `p` threads.

6. OpenMP program that computes the euclidean norm of a float vector, spanning a maximum of `p` threads.
Suggestion: It must not fail if the euclidean norm is smaller than the greatest positive number that can be represented using _floating_ simple precision.

7. OpenMP program that computes the sum of two float vectors, one of them multiplied by an `alpha` constant (saxpy operation), spanning a maximum of `p` threads.

8. OpenMP program that computes the integral, between `a` and `b`, of a polynomical function of grade `m`, splitting the interval `[a, b]` in `n` subintervals and spanning a maximum of `p` threads.

9. OpenMP program that multiplies two long integers stored as a couple of vectors, spanning a maximum of `p` threads.

10. OpenMP program that computes the matrix of minimum distances in a directed graph using Floyd's algorithm.
