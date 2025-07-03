#include <arm_neon.h>

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
        float64x2_t a0 = vdupq_n_f64(A[0 * lda + l]);
        float64x2_t a1 = vdupq_n_f64(A[1 * lda + l]);
        float64x2_t a2 = vdupq_n_f64(A[2 * lda + l]);
        float64x2_t a3 = vdupq_n_f64(A[3 * lda + l]);

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
