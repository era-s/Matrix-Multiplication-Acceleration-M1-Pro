/**********************************************************************
 *  Minimal self-check harness for neon_kernel_4x8
 *
 *  Compile (Apple M-시리즈 예시)
 *      clang++ -O3 -std=c++17 -march=armv8-a+simd -o test_neon_gemm test_neon_gemm.cpp
 *  실행
 *      ./test_neon_gemm
 *********************************************************************/

#include <arm_neon.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//────────────────────── 사용자 매크로/글로벌 그대로 ──────────────────────
#define CACHELINE 64
#define ALIGN(x) __attribute__((aligned(x)))

#define NR 4   // micro-kernel rows
#define MR 8   // micro-kernel cols
#define PC 192   // kc for 테스트 (원본의 KC 대신)
#define NC 1408   // nc for 테스트
#define MC 256   // mc for 테스트 (MR 보다 크거나 같아야 함)

ALIGN(CACHELINE) static double Apanel[PC * NC];
ALIGN(CACHELINE) static double Bpanel[PC * MC];

//───────────────── util: pretty print helpers ─────────────────
void print_mat(const char *name,
               const double *M, int rows, int cols, int ld)
{
    printf("%s (%d×%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%6.1f ", M[i * ld + j]);
        puts("");
    }
    puts("");
}

void print_Apanel(int kc, int nc)
{
    printf("Apanel (kc=%d, nc=%d):\n", kc, nc);
    for (int l = 0; l < kc; ++l) {
        for (int r = 0; r < nc; ++r)
            printf("%6.1f ", Apanel[l * NC + r]);
        puts("");
    }
    puts("");
}

void print_Bpanel(int kc, int mc)
{
    printf("Bpanel (kc=%d, mc=%d):\n", kc, mc);
    for (int l = 0; l < kc; ++l) {
        for (int c = 0; c < mc; ++c)
            printf("%6.1f ", Bpanel[l * MC + c]);
        puts("");
    }
    puts("");
}

//───────────────────── NEON micro-kernel (원본) ─────────────────────
static inline void neon_kernel_4x8(const int P, const double *__restrict A,
                                   const int lda,
                                   const double *__restrict B, const int ldb,
                                   double *__restrict C, const int ldc)
{
    float64x2_t c00 = vld1q_f64(C + 0 * ldc);       float64x2_t c01 = vld1q_f64(C + 0 * ldc + 2);
    float64x2_t c02 = vld1q_f64(C + 0 * ldc + 4);   float64x2_t c03 = vld1q_f64(C + 0 * ldc + 6);

    float64x2_t c10 = vld1q_f64(C + 1 * ldc);       float64x2_t c11 = vld1q_f64(C + 1 * ldc + 2);
    float64x2_t c12 = vld1q_f64(C + 1 * ldc + 4);   float64x2_t c13 = vld1q_f64(C + 1 * ldc + 6);

    float64x2_t c20 = vld1q_f64(C + 2 * ldc);       float64x2_t c21 = vld1q_f64(C + 2 * ldc + 2);
    float64x2_t c22 = vld1q_f64(C + 2 * ldc + 4);   float64x2_t c23 = vld1q_f64(C + 2 * ldc + 6);

    float64x2_t c30 = vld1q_f64(C + 3 * ldc);       float64x2_t c31 = vld1q_f64(C + 3 * ldc + 2);
    float64x2_t c32 = vld1q_f64(C + 3 * ldc + 4);   float64x2_t c33 = vld1q_f64(C + 3 * ldc + 6);

    for (int l = 0; l < P; ++l) {
        float64x2_t a0 = vld1q_dup_f64(A + l * lda + 0);
        float64x2_t a1 = vld1q_dup_f64(A + l * lda + 1);
        float64x2_t a2 = vld1q_dup_f64(A + l * lda + 2);
        float64x2_t a3 = vld1q_dup_f64(A + l * lda + 3);

        float64x2_t b0 = vld1q_f64(B + l * ldb + 0);
        float64x2_t b1 = vld1q_f64(B + l * ldb + 2);
        float64x2_t b2 = vld1q_f64(B + l * ldb + 4);
        float64x2_t b3 = vld1q_f64(B + l * ldb + 6);

        c00 = vfmaq_f64(c00, a0, b0); c01 = vfmaq_f64(c01, a0, b1);
        c02 = vfmaq_f64(c02, a0, b2); c03 = vfmaq_f64(c03, a0, b3);

        c10 = vfmaq_f64(c10, a1, b0); c11 = vfmaq_f64(c11, a1, b1);
        c12 = vfmaq_f64(c12, a1, b2); c13 = vfmaq_f64(c13, a1, b3);

        c20 = vfmaq_f64(c20, a2, b0); c21 = vfmaq_f64(c21, a2, b1);
        c22 = vfmaq_f64(c22, a2, b2); c23 = vfmaq_f64(c23, a2, b3);

        c30 = vfmaq_f64(c30, a3, b0); c31 = vfmaq_f64(c31, a3, b1);
        c32 = vfmaq_f64(c32, a3, b2); c33 = vfmaq_f64(c33, a3, b3);
    }

    vst1q_f64(C + 0 * ldc, c00);        vst1q_f64(C + 0 * ldc + 2, c01);
    vst1q_f64(C + 0 * ldc + 4, c02);    vst1q_f64(C + 0 * ldc + 6, c03);

    vst1q_f64(C + 1 * ldc, c10);        vst1q_f64(C + 1 * ldc + 2, c11);
    vst1q_f64(C + 1 * ldc + 4, c12);    vst1q_f64(C + 1 * ldc + 6, c13);

    vst1q_f64(C + 2 * ldc, c20);        vst1q_f64(C + 2 * ldc + 2, c21);
    vst1q_f64(C + 2 * ldc + 4, c22);    vst1q_f64(C + 2 * ldc + 6, c23);

    vst1q_f64(C + 3 * ldc, c30);        vst1q_f64(C + 3 * ldc + 2, c31);
    vst1q_f64(C + 3 * ldc + 4, c32);    vst1q_f64(C + 3 * ldc + 6, c33);
}

void matmul_neon_kernel_launcher(const int n, const int p, const int m,
                                 const double *__restrict A, const int lda,
                                 const double *__restrict B, const int ldb,
                                 double *__restrict C, const int ldc)
{
    for (int i = 0; i < n; i += NC) {          // ── ① 행 패널 (NC)
        int nc = std::min(NC, n - i);

        for (int j = 0; j < m; j += MC) {      // ── ② 열 패널 (MC)
            int mc = std::min(MC, m - j);
            double *Cpanel = &C[i * ldc + j];

            for (int k = 0; k < p; k += PC) {  // ── ③ KC-슬라이스
                int kc = std::min(PC, p - k);

                /* ---------- 3-A.  B패널 패킹 (kc × MC, 전치 포함) ---------- */
                for (int jr = 0; jr < mc; jr += MR) {
                    int mr = std::min(MR, mc - jr);
                    for (int l = 0; l < kc; ++l) {
                        for (int jj = 0; jj < MR; ++jj) {
                            if (jj < mr)
                                Bpanel[l * MC + jr + jj] =
                                    B[(k + l) * ldb + (j + jr + jj)];
                            else
                                Bpanel[l * MC + jr + jj] = 0.0;
                        }
                    }
                }

                /* ---------- 3-B.  A패널 패킹 (NC × kc) ---------- */
                for (int ir = 0; ir < nc; ir += NR) {
                    int nr = std::min(NR, nc - ir);
                    for (int l = 0; l < kc; ++l) {
                        for (int ii = 0; ii < NR; ++ii) {
                            if (ii < nr)
                                Apanel[l * NC + (ir + ii)] =
                                    A[(i + ir + ii) * lda + k + l];
                            else
                                Apanel[l * NC + (ir + ii)] = 0.0;
                        }
                    }
                }

                /* ---------- 3-C.  마이크로커널 호출 ---------- */
                for (int ir = 0; ir < nc; ir += NR) {
                    double *Crow_ir = Cpanel + ir * ldc;
                    for (int jr = 0; jr < mc; jr += MR) {
                        double *Cblk = Crow_ir + jr;
                        neon_kernel_4x8(kc,
                                        &Apanel[ir], NC,     // A패널: kc가 빠른 축
                                        &Bpanel[jr], MC,     // B패널: kc가 빠른 축
                                        Cblk, ldc);
                    }
                }
            } /* k-loop */
        } /* j-loop */
    } /* i-loop */
}

//──────────────────────────── main ────────────────────────────
int main()
{
    constexpr int N = 5, P = 5, M = 5;   // 4×4 테스트

    // A = [1 2 … 16]
    double A[N][P];
    for (int i = 0, v = 1; i < N; ++i)
        for (int j = 0; j < P; ++j, ++v)
            A[i][j] = static_cast<double>(v);

    // B = I
    double B[P][M] = {};
    for (int i = 0; i < P; ++i)
        B[i][i] = 1.0;

    double C[N][M] = {};   // 출력

    puts("===== INPUT =====");
    print_mat("A", &A[0][0], N, P, P);
    print_mat("B (Identity)", &B[0][0], P, M, M);

    // 계산
    matmul_neon_kernel_launcher(N, P, M, &A[0][0], P,
                                &B[0][0], M,
                                &C[0][0], M);

    puts("===== OUTPUT =====");
    print_mat("C = A × I ?", &C[0][0], N, M, M);

    // 검증
    bool ok = true;
    for (int i = 0; i < N && ok; ++i)
        for (int j = 0; j < M; ++j)
            if (C[i][j] != A[i][j])
                ok = false;
    puts(ok ? "✅  PASS: C == A" : "❌  FAIL: 결과가 A와 다릅니다!");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}