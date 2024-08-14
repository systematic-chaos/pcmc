#ifndef CTIMER
#define CTIMER

    // Avoid C++'s function mangling when including and linking this C timer library
    #ifdef __cplusplus
        extern "C" {
    #endif

            void ctimer(double* elapsed, double* ucpu, double* scpu);

    #ifdef __cplusplus
        }
    #endif

#endif