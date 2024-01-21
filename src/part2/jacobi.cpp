/* jacobi.cpp - Poisson problem in 3d
 *
 */
#include "jacobi.h"

int jacobi(double ***u_curr, double ***u_prev, double ***f, int N,
           int max_iterations) {
            
  int iter = 0;
  double delta = 2.0 / (N - 1);
  double fraction = (1.0 / 6);
  double delta_2 = pow2(delta);
  
  // Put the map outside the inner loops for better performance
  // Jacobi with simple map
  // teams distribute parallel
  // Multiprocessors = 114
  // Maximum number of threads per block = 1024 
  // thread_limit =
  // num_teams = 114 (as multiprocessors)
  // 
  // Collapse (3) --> 3 nested loops
  // Try rounding up to 128 for oversubscription to hide latency
  // 1024 -> thread limit per team (try 512 as well)
  #pragma omp target data map(to: u_prev[0:N][0:N][0:N], f[0:N][0:N][0:N]) map(tofrom: u_curr[0:N][0:N][0:N])
  for (iter = 0; iter < max_iterations; iter++) { 
    #pragma omp target teams distribute parallel for num_teams(_NUM_TEAMS) thread_limit(_THREAD_LIMIT) collapse(3)
    for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        for (int k = 1; k < N - 1; k++) {
          u_curr[i][j][k] =
              fraction *
              (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] +
               u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] +
               delta_2 * f[i][j][k]);
        }
    // Swap pointers on the host loop
    double ***tmp = u_prev;           
    u_prev = u_curr;
    u_curr = tmp;
  }

  // check the odd/even of iterations
  if (iter % 2 == 0) {
    double ***tmp = u_prev;           
    u_prev = u_curr;
    u_curr = tmp;
  }  

  return iter;

}