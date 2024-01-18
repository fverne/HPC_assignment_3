extern "C" {
    #include <cblas.h>
    #include <omp.h>
    #include <stdio.h>
    #include <cublas_v2.h>

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
        // double t1, t2;
        // t1 = omp_get_wtime();
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                C[i][j] = 0;

        #pragma omp target teams distribute parallel for \
        map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
        num_teams(_TEAMS) thread_limit(_THREADS) collapse(2)
        for(int i=0;i<m;i++){
            for (int j=0;j<n;j++){
                double sum = 0;
                for(int l=0;l<k;l++)
                    sum += A[i][l]*B[l][j];
                C[i][j] = sum;
            }
        }
    }

    void 
    matmult_blk_offload(int m, int n, int k, double **A,double **B, double **C) {
        for(int bi = 0; bi < m; bi++) 
            for(int bj = 0; bj < n; bj++) 
                C[bi][bj] = 0;
        
        #pragma omp target teams loop \
        map (to: A[0:m][0:k], B[0:k][0:n], m,k,n) map(tofrom: C[0:m][0:n]) \
        num_teams(_TEAMS) thread_limit(_THREADS) collapse(2)
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
        #pragma omp target enter data map(alloc: A[0:m][0:k], B[0:k][0:n], C[0:m][0:n])
        #pragma omp target update to(B[0:k][0:n]) 
        
        #pragma omp parallel for // parallel for each slab
        for (int s = 0; s < _SLABS; ++s) {
            int slab_len = m / _SLABS; 
            int begin = s * slab_len;

            #pragma omp target update to(A[begin:slab_len][0:k]) depend(out:A) nowait
            #pragma omp target teams distribute parallel for \
            map(to: A[begin:slab_len][:k]) \
            num_teams(_TEAMS) thread_limit(_THREADS)\
            depend(in:A) depend(out:C) nowait \
            collapse(2)
            // apply to mnk
            for(int i = begin; i < begin + slab_len; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for(int l = 0; l < k; l++)
                        sum += A[i][l]*B[l][j];
                    C[i][j] = sum;
                }
            }
            #pragma omp target update from(C[begin:slab_len][:n]) depend(in:C) nowait
        } 
        #pragma omp taskwait // wait on the completion of child tasks
        #pragma omp target exit data map(delete: A[0:m][0:k], B[0:k][0:n], C[0:m][0:n])
    }

    void 
    matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C) {

    }
}