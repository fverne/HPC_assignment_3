/* jacobi_dist.cpp - Poisson problem in 3d
 *
 */
#include "jacobi_dist.h"

int jacobi_dist(double ***u_curr, double ***u_prev, double ***f, int N,
                int max_iterations, double tolerance)
{
  int iter = 0;
  double delta = 2.0 / (N - 1);
  double fraction = (1.0 / 6);
  double delta_2 = pow2(delta);
  double distance;

  #pragma omp target data map(to : u_prev[0 : N][0 : N][0 : N], f[0 : N][0 : N][0 : N]) map(tofrom : u_curr[0 : N][0 : N][0 : N])
  do {
    distance = 0;
    // Check if 128 + 1024 is optimal
    #pragma omp target teams distribute parallel for num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) reduction(+ : distance) collapse(3)
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        for (int k = 1; k < N - 1; k++) {
          u_curr[i][j][k] =
              fraction *
              (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
               u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
               delta_2 * f[i][j][k]);
          // distance
          distance += pow2(u_curr[i][j][k] - u_prev[i][j][k]);
        }
      }
    }

    double ***tmp = u_prev;
    u_prev = u_curr;
    u_curr = tmp;

    ++iter;
    distance = sqrt(distance);

  } while (iter < max_iterations && distance > tolerance);

  // Check the odd/even of iterations
  if (iter % 2 == 0) {
    double ***tmp = u_prev;
    u_prev = u_curr;
    u_curr = tmp;
  }

  return iter;
}