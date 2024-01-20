/* jacobi_dup.cpp - Poisson problem in 3d
 *
 */
#include "util.h"
#include <math.h>

int jacobi_dup(double ***u_curr, double ***u_prev, double ***f, int N, int max_iterations)
{

  int iter = 0;
  double delta = 2.0 / (N - 1);
  double fraction = (1.0 / 6);
  double delta_2 = pow2(delta);

  // device(0) and device(1)
  for (iter = 0; iter < max_iterations; iter++)
  {
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_prev, u_curr, f) num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) device(0)
    for (int i = 1; i < (N - 1) / 2; i++)
      for (int j = 1; j < (N - 1) / 2; j++)
        for (int k = 1; k < (N - 1) / 2; k++)
        {
          u_curr[i][j][k] =
              fraction *
              (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
               u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
               delta_2 * f[i][j][k]);
        }
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_prev, u_curr, f) num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) device(1)
    for (int i = (N - 1) / 2; i < N - 1; i++)
      for (int j = (N - 1) / 2; j < N - 1; j++)
        for (int k = (N - 1) / 2; k < N - 1; k++)
        {
          u_curr[i][j][k] =
              fraction *
              (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
               u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
               delta_2 * f[i][j][k]);
        }
    // Swap pointers on the host
    // Wait for both gpu tasks to finish
    #pragma omp taskwait

    double ***tmp = u_prev;
    u_prev = u_curr;
    u_curr = tmp;

  }

  // Check the odd / even of iterations
  if (iter % 2 == 0)
  {
    double ***tmp = u_prev;
    u_prev = u_curr;
    u_curr = tmp;
  }

  return iter;
}