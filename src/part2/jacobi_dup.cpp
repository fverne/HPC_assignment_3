/* jacobi_dup.cpp - Poisson problem in 3d
 *
 */
#include "util.h"
#include <math.h>

int jacobi_dup(double ***u_curr_0, double ***u_prev_0, double ***f_0, double ***u_curr_1, double ***u_prev_1, double ***f_1, int N, int max_iterations)
{

  int iter = 0;
  double delta = 2.0 / (N - 1);
  double fraction = (1.0 / 6);
  double delta_2 = pow2(delta);

  // device(0) and device(1)
  for (iter = 0; iter < max_iterations; iter++)
  {
    cudaSetDevice(0);
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_prev_0, u_curr_0, f_0) num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) device(0) nowait 
    for (int i = 1; i < (N - 1) / 2; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++)
        {
          u_curr_0[i][j][k] =
              fraction *
              (u_prev_0[i - 1][j][k] + u_prev_0[i + 1][j][k] + u_prev_0[i][j - 1][k] +
               u_prev_0[i][j + 1][k] + u_prev_0[i][j][k - 1] + u_prev_0[i][j][k + 1] +
               delta_2 * f_0[i][j][k]);
        }

    cudaSetDevice(1);
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_prev_1, u_curr_1, f_1) num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) device(1) nowait 
    for (int i = (N - 1) / 2; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++)
        {
          u_curr_1[i][j][k] =
              fraction *
              (u_prev_1[i - 1][j][k] + u_prev_1[i + 1][j][k] + u_prev_1[i][j - 1][k] +
               u_prev_1[i][j + 1][k] + u_prev_1[i][j][k - 1] + u_prev_1[i][j][k + 1] +
               delta_2 * f_1[i][j][k]);
        }
    // Swap pointers on the host
    // Wait for both gpu tasks to finish
    #pragma omp taskwait

    double ***tmp_0 = u_prev_0;
    u_prev_0 = u_curr_0;
    u_curr_0 = tmp_0;

    double ***tmp_1 = u_prev_1;
    u_prev_1 = u_curr_1;
    u_curr_1 = tmp_1; 

  }

  // Check the odd / even of iterations
  if (iter % 2 == 0)
  {
    double ***tmp_0 = u_prev_0;
    u_prev_0 = u_curr_0;
    u_curr_0 = tmp_0;

    double ***tmp_1 = u_prev_1;
    u_prev_1 = u_curr_1;
    u_curr_1 = tmp_1;
  }

  return iter;
}