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

#ifdef _JACOBI_DUP
#include "jacobi_dup.h"
#endif

#define N_DEFAULT 40

void solve_base () {
  // this is the base case
}

void solve_omp() {
  // different output_profile
  
}

void solve_expand() {
  // different output profile
}

void solve_dist(double ***u_curr, double ***u_prev, double ***f, int N, int iter_max), double tolerance) {
  int dev_num;
  int dev_available;
  int dev_init;
  // different output profile
  // also number of iterations 
}

void solve_alloc() {
  // different output profile
  // alloc to device
  // copy back to device
  // freeing device
}

void solve_dup(double ***u_curr, double ***u_prev, double ***f, int N, int iter_max) {
  // different output profile
  // enable cuda peer as well 
  // alloc to two devices
  // copy back to device
  // freeing device
  // disable cuda peer
  // device (used for mappings)
  int dev_num;
  int dev_available;
  int dev_init;

  double *a_prev = NULL; 
  double *a_curr = NULL;
  double *a_f = NULL;
  // host
  double ***u_curr_values = NULL;
  double ***u_prev_values = NULL;
  double ***f_values = NULL;

  // Initialize device sharing
  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);

  iter = jacobi_dup(u_curr, u_prev, f, N, iter_max);

  // Disable cuda peer-to-peer access and offload
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(0);

}

void main(int argc, char *argv[]) {

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
  int dev_available;
  int dev_init;
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
    
  #ifdef _JACOBI
  solve_base();
  #endif

  #ifdef _JACOBI_ALLOC
  solve_alloc();
  #endif

  #ifdef _JACOBI_OMP
  solve_omp();
  #endif

  #ifdef _JACOBI_EXPAND
  solve_expand();
  #endif

  #ifdef _JACOBI_DIST
  solve_dist();
  #endif

  #ifdef _JACOBI_DUP
  solve_dup();
  #endif
  // This will be moved out of the function (in main)
  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);
  
  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;
  /* Allocate memory on device */
  std::cout << "Log: Allocating on device..." << std::endl;
  if ((u_curr = malloc_3d_device(N, N, N, &a_curr, 0)) == NULL) {
      std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
      exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d_device(N, N, N, &a_prev, 0)) == NULL) {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d_device(N, N, N, &a_f, 0)) == NULL) {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  if ((u_curr = malloc_3d_device(N, N, N, &a_curr, 1)) == NULL) {
      std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
      exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d_device(N, N, N, &a_prev, 1)) == NULL) {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d_device(N, N, N, &a_f, 1)) == NULL) {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

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

  // This is temporary
  output_prefix = "jacobi_dup";
  /* Initialize the arrays */
  initialize_u(u_curr_values, N, start_T);
  initialize_u(u_prev_values, N, start_T);
  initialize_f(f_values, N);

  // Transfer values with omp_target_memcpy and free data afterwards 
  // Function signature:
  // omp_target_memcpy(a_curr, u_curr_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  // omp_target_memcpy(a_prev, u_prev_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  // omp_target_memcpy(a_f, f_values, N * N * N * sizeof(double), 0, 0, omp_get_initial_device(), dev_num);
  // // We care about a


  // omp_target_memcpy(void *dst, void *src, size_t length, size_t dst_offset, size_t src_offset, int dst_dev_num, int src_dev_num);
  std::cout << "Log: Allocating memory on device 0 ..." << std::endl;
  omp_target_memcpy(a_curr, u_curr_values, ((N * N * N) / 2) * sizeof(double), 0, 0, dev_init, 0);
  omp_target_memcpy(a_prev, u_prev_values, ((N * N * N) / 2) * sizeof(double), 0, 0, dev_init, 0);
  omp_target_memcpy(a_f, f_values, ((N * N * N) / 2) * sizeof(double), 0, 0, dev_init, 0);
  
  std::cout << "Log: Allocating memory on device 1 ..." << std::endl;
  // omp_target_memcpy(a_curr, u_curr_values, ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  // omp_target_memcpy(a_prev, u_prev_values, ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  // omp_target_memcpy(a_f, f_values, ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  
  // Does it matter to performance that more is allocated?
  omp_target_memcpy(a_curr, u_curr_values, ((N * N * N) / 2) * sizeof(double), 0, ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  omp_target_memcpy(a_prev, u_prev_values, ((N * N * N) / 2) * sizeof(double), 0, ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  omp_target_memcpy(a_f, f_values, ((N * N * N) / 2) * sizeof(double), 0, ((N * N * N) / 2) * sizeof(double), dev_init, 1);
  

  // Freeing host
  std::cout << "Log: Freeing values on host..." << std::endl;
  free_3d(u_curr_values);
  free_3d(u_prev_values);
  free_3d(f_values);
  std::cout << "Log: Freeing values on host finished..." << std::endl;

  itime = omp_get_wtime();

  // CALL JACOBI HERE
  #ifdef _JACOBI_DUP
  iter = jacobi_dup(u_curr, u_prev, f, N, iter_max);
  #endif // _JACOBI_DUP

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  // std::cout << "Log: Freeing values on device..." << std::endl;
  // free_3d_device(4, f, a_curr, a_prev, a_f);
  // std::cout << "Log: Freeing values on device finished..." << std::endl;


  std::cout << "=====================Info=====================" << std::endl;
  printf("N:\t\t\t\t\t%d\n", N);
  printf("Tolerance:\t\t\t\t%f\n", tolerance);
  printf("Execution time:\t\t\t\t%lf\n", exec_time);
  printf("Number of iterations:\t\t\t%d\n", iter);
  printf("Number of iterations per second:\t%f\n", iter / exec_time);
  // This is only for the gpu version
  // printf("Number of threads:\t%d\n", omp_get_max_threads());
  
  // Copy u_curr back to host

  // Disable cuda peer-to-peer access and offload
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(0);
  // switch (output_type) {
  //   case 0:
  //     break;
  //   case 3:
  //     output_ext = ".bin";
  //     sprintf(output_filename, "%s_N%d_T%.8f_I%d%s", output_prefix, N, tolerance,
  //             iter_max, output_ext);
  //     fprintf(stderr, "Wrote binary dump to %s\n.", output_filename);
  //     print_binary(output_filename, N, u_curr);
  //     break;
  //   case 4:
  //     output_ext = ".vtk";
  //     sprintf(output_filename, "%s_N%d_T%.8f_I%d%s", output_prefix, N, tolerance,
  //             iter_max, output_ext);
  //     fprintf(stderr, "Wrote VTK file to %s.\n", output_filename);
  //     print_vtk(output_filename, N, u_curr);
  //     break;
  //   default:
  //     fprintf(stderr, "Non-supported output type!\n");
  //     break;
  //   }
}