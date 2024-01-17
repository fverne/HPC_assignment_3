extern "C" {
    #include <cblas.h>
    #include <omp.h>
    #include <stdio.h>

    void 
    matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C) {
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        #pragma omp parallel for default(none) shared(m, n, k, A, B, C)
        for(int i = 0; i < m; i++) {
            for(int l = 0; l < k; l++) {
                double sum = 0;
                for(int j = 0; j < n; j++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][l] = sum;
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

}