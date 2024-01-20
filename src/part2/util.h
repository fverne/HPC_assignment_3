#ifndef UTIL_H
#define UTIL_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Default values for num_teams and threads according to the hardware

#ifndef _NUM_TEAMS
#define _NUM_TEAMS 128
#endif

#ifndef _THREAD_LIMIT
#define _THREAD_LIMIT 1024
#endif

#define pow2(x) (x) * (x)
#define f(x, y, z)                                                             \
  ((-1 <= x && x <= (-3.0 / 8) && -1 <= y && y <= (-1.0 / 2) &&                \
    (-2.0 / 3) <= z && z <= 0)                                                 \
       ? 200                                                                   \
       : 0)

void initialize_f(double ***f, int N);
void initialize_u(double ***u, int N, int start_T);
double u_value(int x, int y, int z, double start_T);

#endif /* UTIL_H */