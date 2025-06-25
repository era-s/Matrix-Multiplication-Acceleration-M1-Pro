#include "gemm.h"

void gemm(float* C, float* A,float* B, int N, int M, int P) {
    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < M; j++) {
            for (auto k = 0; k < P; k++) {
                C[i * M + j] += A[i * P + k] * B[k * M + j];
            }
        }
    }
}

void gemm_reordered(float* C, float* A,float* B, int N, int M, int P) {
    for (auto i = 0; i < N; i++) {
        for (auto k = 0; k < P; k++) {
            auto temp = A[i * P + k];
            for (auto j = 0; j < M; j++) {
                C[i * M + j] +=  temp * B[k * M + j];
            }
        }
    }
}

void gemm_unroll(float* C, float* A,float* B, int N, int M, int P) {
    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < M; j+=4) {
            for (auto k = 0; k < P; k++) {
                C[i * M + j] += A[i * P + k] * B[k * M + j];
                C[i * M + j + 1] += A[i * P + k] * B[k * M + j + 1];
                C[i * M + j + 2] += A[i * P + k] * B[k * M + j + 2];
                C[i * M + j + 3] += A[i * P + k] * B[k * M + j + 3];
            }
        }
    }
}

void gemm_reordered_unroll(float* C, float* A,float* B, int N, int M, int P) {
    for (auto i = 0; i < N; i++) {
        for (auto k = 0; k < P; k++) {
            auto temp = A[i * P + k];
            for (auto j = 0; j < M; j+=4) {
                C[i * M + j] +=  temp * B[k * M + j];
                C[i * M + j + 1] +=  temp * B[k * M + j + 1];
                C[i * M + j + 2] +=  temp * B[k * M + j + 2];
                C[i * M + j + 3] +=  temp * B[k * M + j + 3];
            }
        }
    }
}
