extern "C" {
    #include <cblas.h>
    #include <omp.h>
    #include <stdio.h>

    void 
    matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C) {
        // initialize C with 0's
        // for(int i = 0; i < m; i++)
        //     for(int j = 0; j < n; j++)
        //         C[i][j] = 0;

        // #pragma omp target parallel
        // {
        //     printf("hello from thread %d\n", omp_get_thread_num());
        // } // end target parallel
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