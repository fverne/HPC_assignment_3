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
#define ITER_DEFAULT 1000


// This just exists peacefully! Do not disturb it!
void warmup() 
{
  int num_devices = omp_get_num_devices();
  std::cout << "Log: Warmup..." << std::endl;
  int sum = 0;
  #pragma omp target teams distribute parallel for reduction(+:sum) collapse(2)
  for (int device_id = 0; device_id < num_devices; ++device_id) {
      for (int i = 0; i < 100; ++i) {
        sum += i;
      }
  }
    std::cout << "Log: Finished warmup..." << std::endl;
}

double ***solve_base(int N, int iter_max, double tolerance, int start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr, N, start_T);
  initialize_u(u_prev, N, start_T);
  initialize_f(f, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  itime = omp_get_wtime();
#ifdef _JACOBI
  iter = jacobi(u_curr, u_prev, f, N, iter_max);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Jacobi (base)" << std::endl;
  std::cout << "Log: Iterations: " << iter << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: Iterations / sec: " << iter / exec_time << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cout << "Log: Number of threads: " << omp_get_max_threads() << std::endl;
  std::cerr << iter / exec_time << std::endl;
  return u_curr;
}

double ***solve_omp(int N, int iter_max, double tolerance, int start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr, N, start_T);
  initialize_u(u_prev, N, start_T);
  initialize_f(f, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  itime = omp_get_wtime();
#ifdef _JACOBI_OMP
  iter = jacobi_omp(u_curr, u_prev, f, N, iter_max);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Jacobi OMP" << std::endl;
  std::cout << "Log: Iterations: " << iter << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: Iterations / sec: " << iter / exec_time << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cout << "Log: Number of threads: " << omp_get_max_threads() << std::endl;
  std::cerr << _NUM_TEAMS << "\t" << _THREAD_LIMIT << "\t" << iter / exec_time << std::endl;
  return u_curr;
}

double ***solve_expand(int N, int iter_max, double tolerance, int start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr, N, start_T);
  initialize_u(u_prev, N, start_T);
  initialize_f(f, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  itime = omp_get_wtime();

#ifdef _JACOBI_EXPAND
  iter = jacobi_expand(u_curr, u_prev, f, N, iter_max, tolerance);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Jacobi Expand" << std::endl;
  std::cout << "Log: Iterations: " << iter << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: Iterations / sec: " << iter / exec_time << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cout << "Log: Number of threads: " << omp_get_max_threads() << std::endl;
  std::cerr << iter / exec_time << std::endl;
  return u_curr;
}

double ***solve_dist(int N, int iter_max, double tolerance, int start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr, N, start_T);
  initialize_u(u_prev, N, start_T);
  initialize_f(f, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  itime = omp_get_wtime();

#ifdef _JACOBI_DIST
  iter = jacobi_dist(u_curr, u_prev, f, N, iter_max, tolerance);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Jacobi GPU norm" << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: Number of iterations: " << iter << std::endl;
  std::cout << "Log: Number of threads: " << omp_get_max_threads() << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cerr << iter / exec_time << std::endl;

  return u_curr;
}

double ***solve_alloc(int N, int iter_max, double start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  double *a_u_curr = NULL;
  double *a_u_prev = NULL;
  double *a_f = NULL;

  // Host
  double ***u_curr_values = NULL;
  double ***u_prev_values = NULL;
  double ***f_values = NULL;

  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocating on device 0
  std::cout << "Log: Allocating on device 0..." << std::endl;
  if ((u_curr = malloc_3d_device(N, N, N, &a_u_curr, 0)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d_device(N, N, N, &a_u_prev, 0)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d_device(N, N, N, &a_f, 0)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocating on device 0 finished..." << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr_values, N, start_T);
  initialize_u(u_prev_values, N, start_T);
  initialize_f(f_values, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  // **Ignore** Function signature:
  // omp_target_memcpy(void *dst, void *src, size_t length, size_t dst_offset, size_t src_offset, int dst_dev_num, int src_dev_num);
  std::cout << "Log: Memcpying on device 0 ..." << std::endl;
  if (omp_target_memcpy(a_u_curr, u_curr_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init) != 0)
  {
    std::cerr << "Error: Memcpy to a_u_curr failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (omp_target_memcpy(a_u_prev, u_prev_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init) != 0)
  {
    std::cerr << "Error: Memcpy to a_u_prev failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (omp_target_memcpy(a_f, f_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init) != 0)
  {
    std::cerr << "Error: Memcpy to a_f failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Log: Finished memcpying on device 0 ..." << std::endl;

  itime = omp_get_wtime();

#ifdef _JACOBI_ALLOC
  iter = jacobi_alloc(u_curr, u_prev, f, N, iter_max);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Copying data back to host" << std::endl;
  omp_target_memcpy(u_curr_values, a_u_curr, (N * N * N) * sizeof(double), 0, 0, dev_init, 0);
  std::cout << "Log: Finished copying data back to host from device 0 ..." << std::endl;

  std::cout << "Log: Freeing device memory ..." << std::endl;
  free_3d_device(2, u_curr, a_u_curr, 0);
  free_3d_device(2, u_prev, a_u_prev, 0);
  free_3d_device(2, f, a_f, 0);
  std::cout << "Log: Freeing device memory finished..." << std::endl;

  std::cout << "Log: Jacobi allocation" << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cerr << iter / exec_time << std::endl;

  return u_curr_values;
}

double ***solve_dup(int N, int iter_max, double start_T)
{
  double ***u_curr = NULL;
  double ***u_prev = NULL;
  double ***f = NULL;

  double itime, ftime, exec_time;
  int dev_num, dev_available, dev_init;
  int iter = 0;

  double *a_u_curr = NULL;
  double *a_u_prev = NULL;
  double *a_f = NULL;

  // Host
  double ***u_curr_values = NULL;
  double ***u_prev_values = NULL;
  double ***f_values = NULL;

  // Initialize device sharing
  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);

  // Getting info on devices
  dev_num = omp_get_default_device();
  dev_available = omp_get_num_devices();
  dev_init = omp_get_initial_device();

  std::cout << "Log: Number of devices: " << dev_available << std::endl;
  std::cout << "Log: Default device: " << dev_num << std::endl;
  std::cout << "Log: Initial device: " << dev_init << std::endl;

  // Allocating on device 0
  std::cout << "Log: Allocating on device 0..." << std::endl;
  if ((u_curr = malloc_3d_device(N, N, N, &a_u_curr, 0)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev = malloc_3d_device(N, N, N, &a_u_prev, 0)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f = malloc_3d_device(N, N, N, &a_f, 0)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocating on device 0 finished..." << std::endl;

  // Allocate memory on host
  std::cout << "Log: Allocating on host..." << std::endl;
  if ((u_curr_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_curr: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((u_prev_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array u_prev: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((f_values = malloc_3d(N, N, N)) == NULL)
  {
    std::cerr << "Error: Array f: allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Log: Allocation finished..." << std::endl;

  // Initialize the arrays
  std::cout << "Log: Initializing arrays on host..." << std::endl;
  initialize_u(u_curr_values, N, start_T);
  initialize_u(u_prev_values, N, start_T);
  initialize_f(f_values, N);
  std::cout << "Log: Initializing arrays on host finished..." << std::endl;

  // **Ignore** Function signature:
  // omp_target_memcpy(void *dst, void *src, size_t length, size_t dst_offset, size_t src_offset, int dst_dev_num, int src_dev_num);
  std::cout << "Log: Memcpying on device 0 ..." << std::endl;
  omp_target_memcpy(a_u_curr, u_curr_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init);
  omp_target_memcpy(a_u_prev, u_prev_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init);
  omp_target_memcpy(a_f, f_values, (N * N * N) * sizeof(double), 0, 0, 0, dev_init);
  std::cout << "Log: Finished memcpying on device 0 ..." << std::endl;

  itime = omp_get_wtime();

#ifdef _JACOBI_DUP
  iter = jacobi_dup(u_curr, u_prev, f, N, iter_max);
#endif

  ftime = omp_get_wtime();
  exec_time = ftime - itime;

  std::cout << "Log: Copying data back to host" << std::endl;
  omp_target_memcpy(u_curr_values, a_u_curr, (N * N * N) * sizeof(double), 0, 0, dev_init, 0);
  std::cout << "Log: Finished copying data back to host from device 0 ..." << std::endl;

  std::cout << "Log: Freeing device memory ..." << std::endl;
  free_3d_device(2, u_curr, a_u_curr, 0);
  free_3d_device(2, u_prev, a_u_prev, 0);
  free_3d_device(2, f, a_f, 0);
  std::cout << "Log: Freeing device memory finished..." << std::endl;

  std::cout << "Log: Disabling peer-to-peer access..." << std::endl;
  // Disable cuda peer-to-peer access and offload
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(0);
  std::cout << "Log: Finished disabling peer-to-peer access..." << std::endl;

  std::cout << "Log: Jacobi dual GPU" << std::endl;
  std::cout << "Log: Iterations: " << iter << std::endl;
  std::cout << "Log: Execution time: " << exec_time << std::endl;
  std::cout << "Log: Iterations / sec: " << iter / exec_time << std::endl;
  std::cout << "Log: N: " << N << std::endl;
  std::cerr << iter / exec_time << std::endl;

  return u_curr_values;
}

void main(int argc, char *argv[])
{

  int N = N_DEFAULT;
  int iter_max = ITER_DEFAULT;
  int iter = 0;
  double tolerance = 0.0;
  double start_T = 0.0;
  int output_type = 0;
  char *output_prefix;
  char *output_ext = "";
  char output_filename[FILENAME_MAX];
  double ***u_curr = NULL;

  // get the parameters from the command line
  N = atoi(argv[1]); // grid size, the number of total
  // grid points in one dimension
  iter_max = atoi(argv[2]);  // max. no. of iterations
  tolerance = atof(argv[3]); // tolerance
  start_T = atof(argv[4]);   // start T for all inner grid points
  if (argc == 6)
    output_type = atoi(argv[5]); // ouput type

#ifdef _JACOBI
  output_prefix = "jacobi";
  // warmup();
  solve_base(N, iter_max, tolerance, start_T);
#endif

#ifdef _JACOBI_ALLOC
  output_prefix = "jacobi_alloc";
  // warmup();
  u_curr = solve_alloc(N, iter_max, start_T);
#endif

#ifdef _JACOBI_OMP
  output_prefix = "jacobi_omp";
  u_curr = solve_omp(N, iter_max, tolerance, start_T);
#endif

#ifdef _JACOBI_EXPAND
  output_prefix = "jacobi_expand";
  u_curr = solve_expand(N, iter_max, tolerance, start_T);
#endif

#ifdef _JACOBI_DIST
  output_prefix = "jacobi_dist";
  // warmup();
  u_curr = solve_dist(N, iter_max, tolerance, start_T);
#endif

#ifdef _JACOBI_DUP
  output_prefix = "jacobi_dup";
  // warmup();
  u_curr = solve_dup(N, iter_max, start_T);
#endif

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
  // }
}