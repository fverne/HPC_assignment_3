/* jacobi_expand.c - Poisson problem in 3d with omp
 *
 */
#include "jacobi_expand.h"

int jacobi_expand(double ***u_curr, double ***u_prev, double ***f,
  int N, int iter_max, double tolerance) {

  int iter = 0;
  double delta = 2.0 / (N - 1);
  double t;
  #pragma omp parallel private(t) 
  {
    do {

      #pragma omp for
      for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
          for (int k = 1; k < N - 1; k++) {
              u_curr[i][j][k] =
                (1.0 / 6) *
                (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
                 u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
                 pow2(delta) * f[i][j][k]);
          }

      // necessary to make sure the loops have finished and the pointer swapping can occur
      #pragma omp barrier
      // one thread again 
      #pragma omp single 
      {
        double ***tmp = u_prev;           
        u_prev = u_curr;
        u_curr = tmp;

        ++iter;
      }
      // implied barrier at the end of `omp single` block
    } while (iter < iter_max);
  }

  if (iter % 2 == 0) {
    double ***tmp = u_prev;
    u_prev = u_curr;
    u_curr = tmp;
  }

  return iter;
}
