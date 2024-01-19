/* main.c - Poisson problem in 3D
 *
 */
#include "alloc3d.h"
#include "print.h"
#include "util.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _JACOBI_ALLOC
#include "jacobi_alloc.h"
#endif

#ifdef _JACOBI_OMP
#include "jacobi_omp.h"
#endif

#ifdef _JACOBI_EXPAND
#include "jacobi_expand.h"
#endif

#ifdef _JACOBI_DIST
#include "jacobi_dist.h"
#endif

#define N_DEFAULT 40

int main(int argc, char *argv[]) {

  int N = N_DEFAULT;
  int iter = 0;
  int iter_max = 1000;
  double tolerance;
  double start_T;
  int output_type = 0;
  char *output_prefix;
  char *output_ext = "";
  char output_filename[FILENAME_MAX];
  // device
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;
  // host
  double ***u_curr_values = NULL;
  double ***u_prev_values = NULL;
  double ***f_values = NULL;
  double itime, ftime, exec_time;
  int dev_num;
  // device (used for mappings)
  double *a_prev = NULL; 
  double *a_curr = NULL;
  double *a_f = NULL;
  /* get the parameters from the command line */
  N = atoi(argv[1]);         /* grid size, the number of total */
                             /* grid points in one dimension */
  iter_max = atoi(argv[2]);  /* max. no. of iterations */
  tolerance = atof(argv[3]); /* tolerance */
  start_T = atof(argv[4]);   /* start T for all inner grid points */
  if (argc == 6) {
    output_type = atoi(argv[5]); /* ouput type */
  }
  
  dev_num = omp_get_default_device();

  #ifdef _JACOBI_ALLOC
  /* Allocate memory on device */
  std::cout << "Log: Allocating on device..." << std::endl;
  if ((u_curr = malloc_3d_device(N, N, N, &a_curr)) == NULL) {
      std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
      exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d_device(N, N, N, &a_prev)) == NULL) {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d_device(N, N, N, &a_f)) == NULL) {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;
  #endif
  /* Allocate memory on host */
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr_values = malloc_3d(N, N, N)) == NULL) {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev_values = malloc_3d(N, N, N)) == NULL) {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f_values = malloc_3d(N, N, N)) == NULL) {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  #ifdef _JACOBI_ALLOC
  output_prefix = "jacobi_alloc";
  /* Initialize the arrays */
  initialize_u(u_curr_values, N, start_T);
  initialize_u(u_prev_values, N, start_T);
  initialize_f(f_values, N);

  // Transfer values with omp_target_memcpy and free data afterwards 
  // Function signature:
  // omp_target_memcpy(void *dst, void *src, size_t length, size_t dst_offset, size_t src_offset, int dst_dev_num, int src_dev_num);
  omp_target_memcpy(a_curr, u_curr_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  omp_target_memcpy(a_prev, u_prev_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  omp_target_memcpy(a_f, f_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  // We care about a

  // Freeing host
  std::cout << "Log: Freeing values on host..." << std::endl;
  free_3d(u_curr_values);
  free_3d(u_prev_values);
  free_3d(f_values);
  std::cout << "Log: Freeing values on host finished..." << std::endl;

  // call jacobi here

  std::cout << "Log: Freeing values on device..." << std::endl;
  free_3d_device(6, u_curr, u_prev, f, a_curr, a_prev, a_f);
  std::cout << "Log: Freeing values on device finished..." << std::endl;
  #endif

  itime = omp_get_wtime();

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "=====================Info=====================" << std::endl;
  printf("N:\t\t\t\t\t%d\n", N);
  printf("Tolerance:\t\t\t\t%f\n", tolerance);
  std::cout << "Time:" << exec_time << std::endl;
  printf("Number of iterations:\t\t\t%d\n", iter);
  printf("Number of iterations per second:\t%f\n", iter / exec_time);
  printf("Number of threads:\t%d\n", omp_get_max_threads());

  switch (output_type) {
  case 0:
    break;
  case 3:
    output_ext = ".bin";
    sprintf(output_filename, "%s_N%d_T%.8f_I%d%s", output_prefix, N, tolerance,
            iter_max, output_ext);
    fprintf(stderr, "Wrote binary dump to %s\n.", output_filename);
    print_binary(output_filename, N, u_curr);
    break;
  case 4:
    output_ext = ".vtk";
    sprintf(output_filename, "%s_N%d_T%.8f_I%d%s", output_prefix, N, tolerance,
            iter_max, output_ext);
    fprintf(stderr, "Wrote VTK file to %s.\n", output_filename);
    print_vtk(output_filename, N, u_curr);
    break;
  default:
    fprintf(stderr, "Non-supported output type!\n");
    break;
  }

  /* De-allocate memory */
  #ifdef _JACOBI_ALLOC
  #endif /* _JACOBI_ALLOC */

  return (0);
}