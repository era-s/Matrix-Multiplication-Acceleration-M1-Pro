#include "gemm.h"
#include "immintrin.h"

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

void gemm_intrinsic(float* C, const float* A, const float* B, int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; j += 8) {  // AVX2는 8개 float씩 처리
            __m256 c = _mm256_loadu_ps(C + i * M + j);  // C[i][j ~ j+7]

            for (int k = 0; k < P; ++k) {
                float a_scalar = A[i * P + k];  // A[i][k]
                __m256 a = _mm256_set1_ps(a_scalar);  // broadcast
                __m256 b = _mm256_loadu_ps(B + k * M + j);  // B[k][j ~ j+7]
                c = _mm256_fmadd_ps(a, b, c);  // FMA: c += a * b
            }

            _mm256_storeu_ps(C + i * M + j, c);  // 결과 저장
        }
    }
}

void gemm_reordered_intrinsic(float* C, const float* A, const float* B, int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < P; ++k) {
            __m256 a = _mm256_set1_ps(A[i * P + k]);  // broadcast A[i][k]
            for (int j = 0; j < M; j += 8) {
                __m256 b = _mm256_loadu_ps(B + k * M + j);       // B[k][j~j+7]
                __m256 c = _mm256_loadu_ps(C + i * M + j);       // C[i][j~j+7]
                c = _mm256_fmadd_ps(a, b, c);                    // FMA: c += a * b
                _mm256_storeu_ps(C + i * M + j, c);              // 저장
            }
        }
    }
}
