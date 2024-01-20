#include <cuda_runtime.h>
#include <cublas_v2.h>
extern "C" {
    #include <cblas.h>
    #include <omp.h>
    #include <stdio.h>

    #ifndef _TEAMS
    #define _TEAMS 114
    #endif

    #ifndef _THREADS
    #define _THREADS 16
    #endif

    #ifndef _BLK_SIZE 
    #define _BLK_SIZE 16
    #endif

    #ifndef _SLABS
    #define _SLABS 4
    #endif

    #define min(a,b) (((a)<(b))?(a):(b))

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
                    // #pragma omp atomic
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
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        #pragma omp target teams distribute parallel for \
        map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
        num_teams(_TEAMS) thread_limit(_THREADS) // you cannot use collapse for mkn, that's why it is not optimal for GPU
        for(int i = 0; i < m; i++){
            for(int l = 0; l < k; l++){
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][l]*B[l][j]; // data race here 
                }
            }
        }

    }

    void 
    matmult_mnk_offload(int m, int n, int k, double **A, double **B, double**C) {
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        double t1, t2, t3;
        t1 = omp_get_wtime();
        // data won't be transfered twice 
        #pragma omp target data map(to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n])
        {
            t2 = omp_get_wtime();
            // TRANSER_TIMING starts
            #pragma omp target teams distribute parallel for \
            map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
            num_teams(_TEAMS) thread_limit(_THREADS) \
            collapse(2)
            // num_teams(_TEAMS) thread_limit(_THREADS) collapse(2)
            for(int i=0; i<m; i++){
                for (int j=0; j<n; j++){
                    double sum = 0;
                    for(int l=0; l<k; l++)
                        sum += A[i][l]*B[l][j];
                    C[i][j] = sum;
                }
            }
        } //end parallel region
        t3 = omp_get_wtime();
        printf("TransTime: %f\tTime: %f\t", 1e3*(t2-t1), 1e3*(t3-t1));
    }

    void 
    matmult_blk_offload(int m, int n, int k, double **A,double **B, double **C) {
        for(int bi = 0; bi < m; bi++) 
            for(int bj = 0; bj < n; bj++) 
                C[bi][bj] = 0;
        
        #pragma omp target teams loop \
        map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
        collapse(2)
        for (int i1 = 0; i1 < m; i1 += _BLK_SIZE) {
            for (int j = 0; j < n; j++) {
                int i2, l;
                if (i1 + _BLK_SIZE - 1 < m) {
                    double sum[_BLK_SIZE] = {0};
                    for(l = 0; l < k; l++) {   
                        for(i2 = 0; i2 < _BLK_SIZE; i2++){
                            sum[i2] += A[i1+i2][l] * B[l][j];
                        }
                    }
                    for (int i = 0; i < _BLK_SIZE; i++){
                        C[i1 + i][j] = sum[i];
                    }
                } else { 
                    for(l = 0;l < k; l++){   
                        for(i2 = 0; i2 < (m - i1); i2++){
                            C[i1+i2][j] += A[i1+i2][l]*B[l][j];
                        }
                    }
                }
            }
        }
    }

    void 
    matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
        double t1, t2;
        t1 = omp_get_wtime();

        #pragma omp parallel for // parallel for each slab
        for (int s = 0; s < _SLABS; ++s) {
            int slab_len = m / _SLABS; 
            int begin = s * slab_len;

            #pragma omp target teams distribute parallel for collapse(2) nowait \
            map (to: A[begin:slab_len][0:k], B[0:k][0:n]) \
            map (from:C[begin:slab_len][0:n])
            for(int i = begin; i < begin + slab_len; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for(int l = 0; l < k; l++)
                        sum += A[i][l]*B[l][j];
                    C[i][j] = sum;
                }
            }
        } 
        #pragma omp taskwait 

        t2 = omp_get_wtime();
        printf("Time: %f\t", 1e3*(t2-t1));
    }

    void 
    matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C) {
       cublasHandle_t handle;
       cublasCreate(&handle);

       double *d_A, *d_B, *d_C;
       cudaMalloc((void **)&d_A, m * k * sizeof(double));
       cudaMalloc((void **)&d_B, k * n * sizeof(double));
       cudaMalloc((void **)&d_C, m * n * sizeof(double));

       cudaMemcpy(d_A, *A, m * k * sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_B, *B, k * n * sizeof(double), cudaMemcpyHostToDevice);

       double alpha = 1.0;
       double beta = 0.0;

       cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

       cudaMemcpy(*C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToDevice);

       cudaFree(d_A);
       cudaFree(d_B);
       cudaFree(d_C);
       cublasDestroy(handle);
    }
}