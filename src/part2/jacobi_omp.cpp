/* jacobi.c - Poisson problem in 3d with omp
 *
 */
#include <math.h>
#include <omp.h>
#include "jacobi_omp.h"
#include "util.h"

// See which version is better from the previous ones (fix the old version)
int jacobi_omp(
  double ***u_curr, double ***u_prev, double ***f, 
  int N, int iter_max) {
  int iter = 0;
  double delta = 2.0 / (N - 1);
  
  for (iter = 0; iter < iter_max; iter++) {
    #pragma omp parallel for 
    for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++) {
          u_curr[i][j][k] =
              (1.0 / 6) *
              (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
               u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
               pow2(delta) * f[i][j][k]);
    }

    double ***tmp = u_prev;           
    u_prev = u_curr;
    u_curr = tmp;
  } 

  // check the odd/even of iterations
  if (iter % 2 == 0) {
    for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++) { 
          u_curr[i][j][k] = u_prev[i][j][k];
        }
  }  

  return iter;
}