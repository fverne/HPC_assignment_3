#include "alloc3d.h"

double ***malloc_3d(int m, int n, int k) {

  if (m <= 0 || n <= 0 || k <= 0)
    return NULL;

  double ***p =
      (double ***)malloc(m * sizeof(double **) + m * n * sizeof(double *));
  if (p == NULL) {
    return NULL;
  }

  for (int i = 0; i < m; i++) {
    p[i] = (double **)p + m + i * n;
  }

  double *a = (double *)malloc(m * n * k * sizeof(double));
  if (a == NULL) {
    free(p);
    return NULL;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      p[i][j] = a + (i * n * k) + (j * k);
    }
  }

  return p;
}

double ***malloc_3d_device(int m, int n, int k, double **a_data) {
  
  int dev_num = omp_get_default_device();
  if (m <= 0 || n <= 0 || k <= 0)
  {
    std::cerr << "Error: Out of range." << std::endl;
    return NULL;
  }

  // allocate array of pointers in device
  double ***p = (double***)omp_target_alloc(m * sizeof(double **) +
                            m * n * sizeof(double *), dev_num);
  if (p == NULL) {
    std::cerr << "Error: Failed to allocate memory using omp_target_alloc." << std::endl;
    return NULL;
  }

  // improve performance on initialization
  #pragma omp target teams distribute parallel for is_device_ptr(p)
  for (int i = 0; i < m; i++)
    p[i] = (double **)p + m + i * n;

  // `a` holds the data values, real numerical values
  double* a = (double *)omp_target_alloc(m * n * k * sizeof(double), dev_num);
  if (a == NULL) 
  {
    free_3d_device(1, p);  
    std::cerr << "Error: Failed to allocate memory using omp_target_alloc." << std::endl;
    return NULL;
  }

  #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(p, a)
    for(int i = 0; i < m; i++) 
      for(int j = 0; j < n; j++) 
        p[i][j] = a + (i * n * k) + (j * k);

  *a_data = a;
  return p;
}

void free_3d(double ***p) {
  free(p[0][0]);
  free(p);
}

void free_3d_device(int argc, ...) {
  va_list argv;
  int dev_num = omp_get_default_device();
  va_start(argv, argc);
  
  for (int i = 0; i < argc; i++) {
      double ***p = va_arg(argv, double***);
      if (p != NULL) {
          #pragma omp target is_device_ptr(p)
          {
              omp_target_free(p, dev_num);
          }
      }
  }
  va_end(args);
}