#include "cblas.h"
#include <math.h>

void
matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void
matmult_lib(int m, int n, int k, double **A, double **B, double**C) {
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, 
    alpha, *A, k, 
    *B, n, beta, *C, n);
}

void 
matmult_mnk(int m,int n,int k,double **A,double **B,double **C) {
    matmult_nat(m, n, k, A, B, C);
}

// **
void 
matmult_mkn(int m,int n,int k,double **A,double **B,double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int i = 0; i < m; i++) {
        for(int l = 0; l < k; l++) {
            for(int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void 
matmult_nmk(int m,int n,int k,double **A,double **B,double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int j = 0; j < n; j++) {
        for(int i = 0; i < m; i++) {
            for(int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void 
matmult_nkm(int m,int n,int k,double **A,double **B,double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int j = 0; j < n; j++) {
        for(int l = 0; l < k; l++) {
            for(int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    } 
}

// **
void 
matmult_kmn(int m,int n,int k,double **A,double **B,double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int l = 0; l < k; l++) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void 
matmult_knm(int m,int n,int k,double **A,double **B,double **C) {
    // initialize C with 0's
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            C[i][j] = 0;

    for(int l = 0; l < k; l++) {
        for(int j = 0; j < n; j++) {
            for(int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void 
matmult_blk(int m,int n,int k,double **A,double **B,double **C, int bs) {
    // initialize C with 0's
    for(int bi=0; bi<m; bi++) 
        for(int bj=0; bj<n; bj++) 
            C[bi][bj] = 0;

    // blocking matrix multiplication
    // mkn
        for(int bi=0; bi<m; bi+=bs) {
            for(int bl=0; bl<k; bl+=bs) {
                for(int bj=0; bj<n; bj+=bs) {
                    for(int i=0; i<fmin(m-bi, bs); i++) {
                        for(int l=0; l<fmin(k-bl, bs); l++) {
                            for(int j=0; j<fmin(n-bj, bs); j++) {
                            C[bi+i][bj+j] += A[bi+i][bl+l]*B[bl+l][bj+j];
                        }
                    }
                }
            }
        }
    }
}