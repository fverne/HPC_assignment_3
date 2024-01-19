#ifndef __ALLOC_3D
#define __ALLOC_3D

#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <cstdarg>

double ***malloc_3d(int m, int n, int k);
double ***malloc_3d_device(int m, int n, int k, double **);

#define HAS_FREE_3D
void free_3d(double ***array3D);
void free_3d_device(int argc, ...);
#endif /* __ALLOC_3D */