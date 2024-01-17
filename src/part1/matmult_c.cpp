extern "C" {
    #include <cblas.h>
    #include <omp.h>
    #include <stdio.h>

    #ifndef _TEAMS
    #define _TEAMS 16384
    #endif

    #ifndef _THREADS
    #define _THREADS 16
    #endif

    void 
    matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C) {
        // double t1, t2;
        // t1 = omp_get_wtime();
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        #pragma omp parallel shared(m, n, k, A, B, C) for
        for(int i = 0; i < m; i++) {
            for(int l = 0; l < k; l++) {
                for(int j = 0; j < n; j++) {
                    #pragma omp atomic
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
        // t2 = omp_get_wtime();
        // printf("Time: %lf\n", (t2-t1)*1e3);
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

    // offload
    void 
    matmult_mkn_offload(int m, int n, int k, double **A, double **B, double**C) {
        // double t1, t2;
        // t1 = omp_get_wtime();
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        #pragma omp target teams distribute parallel for \
        map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
        num_teams(_TEAMS) thread_limit(_THREADS) collapse(2)
        for(int i=0;i<m;i++){
            for(int l=0;l<k;l++){
                double sum = 0;
                for (int j=0;j<n;j++)
                    sum += A[i][j]*B[j][l];
                C[i][l] = sum;  
            }
        }
        // t2 = omp_get_wtime();
        // printf("Time with transfer: %f\n", 1e3*(t2-t1));
    }

    void 
    matmult_mnk_offload(int m, int n, int k, double **A, double **B, double**C) {

    }
}