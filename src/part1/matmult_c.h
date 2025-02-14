#ifndef __MATMULT_H
#define __MATMULT_H

void matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C);

void matmult_lib(int m,int n,int k,double **A,double **B,double **C);

void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double**C);

void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double**C);

void matmult_blk_offload(int m, int n, int k, double **A,double **B, double **C);

void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C);

void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C);
#endif