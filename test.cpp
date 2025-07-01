void rank_one_kernel_4x4(int k, 
                         double* C, int ldc, 
                         const double* A, int lda, 
                         const double* B, int ldb) {
    double c00 = C[0 + 0 * ldc];
    double c01 = C[0 + 1 * ldc];
    double c02 = C[0 + 2 * ldc];
    double c03 = C[0 + 3 * ldc];

    double c10 = C[1 + 0 * ldc];
    double c11 = C[1 + 1 * ldc];
    double c12 = C[1 + 2 * ldc];
    double c13 = C[1 + 3 * ldc];

    double c20 = C[2 + 0 * ldc];
    double c21 = C[2 + 1 * ldc];
    double c22 = C[2 + 2 * ldc];
    double c23 = C[2 + 3 * ldc];

    double c30 = C[3 + 0 * ldc];
    double c31 = C[3 + 1 * ldc];
    double c32 = C[3 + 2 * ldc];
    double c33 = C[3 + 3 * ldc];

    for (auto l = 0; l < k; l++) {
        double a0 = A[0 + l * lda];
        double a1 = A[1 + l * lda];
        double a2 = A[2 + l * lda];
        double a3 = A[3 + l * lda];

        double b0 = B[l + 0 * ldb];
        double b1 = B[l + 1 * ldb];
        double b2 = B[l + 2 * ldb];
        double b3 = B[l + 3 * ldb];

        c00 += a0 * b0;   c01 += a0 * b1;
        c02 += a0 * b2;   c03 += a0 * b3;

        c10 += a1 * b0;   c11 += a1 * b1;
        c12 += a1 * b2;   c13 += a1 * b3;

        c20 += a2 * b0;   c21 += a2 * b1;
        c22 += a2 * b2;   c23 += a2 * b3;

        c30 += a3 * b0;   c31 += a3 * b1;
        c32 += a3 * b2;   c33 += a3 * b3;
    }

    C[0 + 0 * ldc] = c00;   C[0 + 1 * ldc] = c01;
    C[0 + 2 * ldc] = c02;   C[0 + 3 * ldc] = c03;

    C[1 + 0 * ldc] = c10;   C[1 + 1 * ldc] = c11;
    C[1 + 2 * ldc] = c12;   C[1 + 3 * ldc] = c13;

    C[2 + 0 * ldc] = c20;   C[2 + 1 * ldc] = c21;
    C[2 + 2 * ldc] = c22;   C[2 + 3 * ldc] = c23;

    C[3 + 0 * ldc] = c30;   C[3 + 1 * ldc] = c31;
    C[3 + 2 * ldc] = c32;   C[3 + 3 * ldc] = c33;
    
}