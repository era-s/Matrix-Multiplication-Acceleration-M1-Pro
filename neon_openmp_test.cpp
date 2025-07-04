#include <arm_neon.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// Define block sizes
#define NR 4
#define MR 4
#define NC 256
#define KC 128
#define MC 384

#define CACHELINE 64
#if defined(__GNUC__) || defined(__clang__)
    #define ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
    #define ALIGN(x) __declspec(align(x))
#else
    #define ALIGN(x)
#endif

ALIGN(CACHELINE) static double Apanel[KC * NR] __attribute__((aligned(4096)));
ALIGN(CACHELINE) static double Bpanel[KC * MR] __attribute__((aligned(4096)));

void neon_kernel_4x4(const int P, const double * __restrict A, const int lda,
                     const double * __restrict B, const int ldb,
                     double * __restrict C, const int ldc) {
    
    // C 로드
    // 똥 neon은 256byte 없음 128 나눠서
    float64x2_t c00 = vld1q_f64(C + 0 * ldc);
    float64x2_t c01 = vld1q_f64(C + 0 * ldc + 2);
    float64x2_t c10 = vld1q_f64(C + 1 * ldc);
    float64x2_t c11 = vld1q_f64(C + 1 * ldc + 2);
    float64x2_t c20 = vld1q_f64(C + 2 * ldc);
    float64x2_t c21 = vld1q_f64(C + 2 * ldc + 2);
    float64x2_t c30 = vld1q_f64(C + 3 * ldc);
    float64x2_t c31 = vld1q_f64(C + 3 * ldc + 2);

    for (auto l = 0; l < P; l++) {
        // A 로드
        float64x2_t a0 = vdupq_n_f64(A[l * lda + 0]);
        float64x2_t a1 = vdupq_n_f64(A[l * lda + 1]);
        float64x2_t a2 = vdupq_n_f64(A[l * lda + 2]);
        float64x2_t a3 = vdupq_n_f64(A[l * lda + 3]);

        // B 로드
        float64x2_t b0 = vld1q_f64(B + l * ldb + 0);
        float64x2_t b1 = vld1q_f64(B + l * ldb + 2);

        // C 업데이트
        c00 = vfmaq_f64(c00, a0, b0);
        c01 = vfmaq_f64(c01, a0, b1);
        c10 = vfmaq_f64(c10, a1, b0);
        c11 = vfmaq_f64(c11, a1, b1);
        c20 = vfmaq_f64(c20, a2, b0);
        c21 = vfmaq_f64(c21, a2, b1);
        c30 = vfmaq_f64(c30, a3, b0);
        c31 = vfmaq_f64(c31, a3, b1);
    }

    // C 저장
    vst1q_f64(C + 0 * ldc, c00);
    vst1q_f64(C + 0 * ldc + 2, c01);
    vst1q_f64(C + 1 * ldc, c10);
    vst1q_f64(C + 1 * ldc + 2, c11);
    vst1q_f64(C + 2 * ldc, c20);
    vst1q_f64(C + 2 * ldc + 2, c21);
    vst1q_f64(C + 3 * ldc, c30);
    vst1q_f64(C + 3 * ldc + 2, c31);
}

inline void pack_blockA(const double * __restrict A, const int lda, const int i,
                        const int p, const int nc, const int kc, double * __restrict Apanel) {
    
}

inline void pack_blockB();

void matmul_neon_openmp_kernel_launcher(const int n, const int p, const int m,
                                        const double * __restrict A, const int lda,
                                        const double * __restrict B, const int ldb,
                                        double * __restrict C, const int ldc) {
    #pragma omp parallel
    {
        ALIGN(CACHELINE) static double Apanel[KC * NR] __attribute__((aligned(4096)));
        ALIGN(CACHELINE) static double Bpanel[KC * MR] __attribute__((aligned(4096)));

        #pragma omp for collapse(2) schedule(static)
        for (auto i = 0; i < n; i += NC) {
            for (auto j = 0; j < m; j += MC) {
            int nc = std::min(NC, n - i);
            int mc = std::min(MC, m - j);
            }
        }
}
