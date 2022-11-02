#include <sys/time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>
#include "ctimer.h"

static double   timetick;
static double   tstart = 0.;
static double   ucpustart = 0.;
static double   scpustart = 0.;
static int      first = 1;

void ctimer(double* elapsed, double* ucpu, double* scpu) {
    struct tms      cpu;
    struct timeval  tp;

    if (first) {

        /** Initialize clock */
        timetick = 1. / (double)(sysconf(_SC_CLK_TCK));
        first = 0;
        gettimeofday(&tp, NULL);
        tstart = (double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6;

        /* Initialize CPU time */
        times(&cpu);
        ucpustart = (double)(cpu.tms_utime + cpu.tms_cutime) * timetick;
        scpustart = (double)(cpu.tms_stime + cpu.tms_cstime) * timetick;

        /* Return values */
        *elapsed = 0.0e0;
        *ucpu = 0.0e0;
        *scpu = 0.0e0;
    } else {

        /* Get clock time */
        gettimeofday(&tp, NULL);
        *elapsed = (double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6 - tstart;

        /* Get CPU time */
        times(&cpu);
        *ucpu = (double)(cpu.tms_utime + cpu.tms_cutime) * timetick - ucpustart;
        *scpu = (double)(cpu.tms_stime + cpu.tms_cstime) * timetick - scpustart;
        first = 1;
    }

    return;
}
