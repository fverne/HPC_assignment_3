#ifndef __MATMULT_H
#define __MATMULT_H

void matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C);

void matmult_lib(int m,int n,int k,double **A,double **B,double **C);

void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double**C);

#endif