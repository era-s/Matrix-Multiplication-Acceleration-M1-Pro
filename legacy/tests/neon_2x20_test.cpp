#include <arm_neon.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

// Define block sizes
#define NR 2
#define MR 20

#define PC 120
#define NC 500
#define MC 120

#define CACHELINE 64
#if defined(__GNUC__) || defined(__clang__)
    #define ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
    #define ALIGN(x) __declspec(align(x))
#else
    #define ALIGN(x)
#endif

// ALIGN(CACHELINE) static double Apanel[PC * NC];
// ALIGN(CACHELINE) static double Bpanel[PC * MC];
ALIGN(CACHELINE) static double Apanel[PC * NC] __attribute__((aligned(16384)));
ALIGN(CACHELINE) static double Bpanel[MC * PC] __attribute__((aligned(16384)));
// ALIGN(CACHELINE) static double Apanel[PC * NC] __attribute__((aligned(4096)));
// ALIGN(CACHELINE) static double Bpanel[PC * MC] __attribute__((aligned(4096)));
// ALIGN(CACHELINE) static double C_temp[NC * MC];

/* 4×8 NEON micro-kernel  —  Row-major
 *  A : MR(4)×kc  col-major (ldA = NR)
 *  B : kc×NR(8)  row-major (ldB = MR)
 *  C : MR×NR     row-major (ldC = 전체 열 m)
 */
static inline void neon_kernel_4x8(const int kc,
                                   const double* __restrict A, const int ldA,
                                   const double* __restrict B, const int ldB,
                                   double*       __restrict C, const int ldC)
{
    /* --- C 4×8 블록 프롤로그 --------------------------------------- */
    float64x2_t c00 = vld1q_f64(C + 0*ldC + 0);   /* 행 0, 열 0-1 */
    float64x2_t c01 = vld1q_f64(C + 0*ldC + 2);   /* 행 0, 열 2-3 */
    float64x2_t c02 = vld1q_f64(C + 0*ldC + 4);
    float64x2_t c03 = vld1q_f64(C + 0*ldC + 6);
    float64x2_t c04 = vld1q_f64(C + 0*ldC + 8);
    float64x2_t c05 = vld1q_f64(C + 0*ldC + 10);   
    float64x2_t c06 = vld1q_f64(C + 0*ldC + 12);
    float64x2_t c07 = vld1q_f64(C + 0*ldC + 14);
    float64x2_t c08 = vld1q_f64(C + 0*ldC + 16);
    float64x2_t c09 = vld1q_f64(C + 0*ldC + 18);

    float64x2_t c10 = vld1q_f64(C + 1*ldC + 0);
    float64x2_t c11 = vld1q_f64(C + 1*ldC + 2);
    float64x2_t c12 = vld1q_f64(C + 1*ldC + 4);
    float64x2_t c13 = vld1q_f64(C + 1*ldC + 6);
    float64x2_t c14 = vld1q_f64(C + 1*ldC + 8);
    float64x2_t c15 = vld1q_f64(C + 1*ldC + 10);
    float64x2_t c16 = vld1q_f64(C + 1*ldC + 12);
    float64x2_t c17 = vld1q_f64(C + 1*ldC + 14);
    float64x2_t c18 = vld1q_f64(C + 1*ldC + 16);
    float64x2_t c19 = vld1q_f64(C + 1*ldC + 18);
    

    /* --- kc-loop : 외적 ------------------------------------------------ */
    for (int l = 0; l < kc; ++l)
    {
        /* A – 각 행 스칼라 broadcast */
        float64x2_t a0 = vld1q_dup_f64(A + l*ldA + 0);
        float64x2_t a1 = vld1q_dup_f64(A + l*ldA + 1);

        /* B – kc 행 한 줄을 8-열(4×2) 로드 */
        float64x2_t b0 = vld1q_f64(B + l*ldB + 0);  
        float64x2_t b1 = vld1q_f64(B + l*ldB + 2);  
        float64x2_t b2 = vld1q_f64(B + l*ldB + 4);  
        float64x2_t b3 = vld1q_f64(B + l*ldB + 6);  
        float64x2_t b4 = vld1q_f64(B + l*ldB + 8);  
        float64x2_t b5 = vld1q_f64(B + l*ldB + 10);  
        float64x2_t b6 = vld1q_f64(B + l*ldB + 12);  
        float64x2_t b7 = vld1q_f64(B + l*ldB + 14);  
        float64x2_t b8 = vld1q_f64(B + l*ldB + 16);  
        float64x2_t b9 = vld1q_f64(B + l*ldB + 18);  
        

        /* FMA : C += A_row × B_col */
        c00 = vfmaq_f64(c00, a0, b0);  c01 = vfmaq_f64(c01, a0, b1);
        c02 = vfmaq_f64(c02, a0, b2);  c03 = vfmaq_f64(c03, a0, b3);
        c04 = vfmaq_f64(c04, a0, b4);  c05 = vfmaq_f64(c05, a0, b5);
        c06 = vfmaq_f64(c06, a0, b6);  c07 = vfmaq_f64(c07, a0, b7);
        c08 = vfmaq_f64(c08, a0, b8);  c09 = vfmaq_f64(c09, a0, b9);

        c10 = vfmaq_f64(c10, a1, b0);  c11 = vfmaq_f64(c11, a1, b1);
        c12 = vfmaq_f64(c12, a1, b2);  c13 = vfmaq_f64(c13, a1, b3);
        c14 = vfmaq_f64(c14, a1, b4);  c15 = vfmaq_f64(c15, a1, b5);
        c16 = vfmaq_f64(c16, a1, b6);  c17 = vfmaq_f64(c17, a1, b7);
        c18 = vfmaq_f64(c18, a1, b8);  c19 = vfmaq_f64(c19, a1, b9);
    }

    /* --- 결과 저장 ----------------------------------------------------- */
    vst1q_f64(C + 0*ldC + 0, c00);  vst1q_f64(C + 0*ldC + 2, c01);
    vst1q_f64(C + 0*ldC + 4, c02);  vst1q_f64(C + 0*ldC + 6, c03);
    vst1q_f64(C + 0*ldC + 8, c04);  vst1q_f64(C + 0*ldC + 10, c05);
    vst1q_f64(C + 0*ldC + 12, c06);  vst1q_f64(C + 0*ldC + 14, c07);
    vst1q_f64(C + 0*ldC + 16, c08);  vst1q_f64(C + 0*ldC + 18, c09);

    vst1q_f64(C + 1*ldC + 0, c10);  vst1q_f64(C + 1*ldC + 2, c11);
    vst1q_f64(C + 1*ldC + 4, c12);  vst1q_f64(C + 1*ldC + 6, c13);
    vst1q_f64(C + 1*ldC + 8, c14);  vst1q_f64(C + 1*ldC + 10, c15);
    vst1q_f64(C + 1*ldC + 12, c16);  vst1q_f64(C + 1*ldC + 14, c17);
    vst1q_f64(C + 1*ldC + 16, c18);  vst1q_f64(C + 1*ldC + 18, c19);
}

/*------------------------------------------------------------*/
/*  j-k-i  blocked GEMM  launcher  (Row-major)                 */
/*  outer-N(j) → KC(k) → MC(i)                                */
/*------------------------------------------------------------*/
void matmul_neon_kernel_launcher(int n,  /* A rows / C rows           */
                                 int p,  /* shared dim (k)            */
                                 int m,  /* B cols  / C cols          */
                                 const double*  A, int lda, /* lda = p */
                                 const double*  B, int ldb, /* ldb = m */
                                 double*        C, int ldc) /* ldc = m */
{
    /* ---- Loop-3 : N-dimension (B,C 열) ------------------------------ */
    for (int j = 0; j < m; j += MC)
    {
        int mc = std::min(MC, m - j);

        /* ---- Loop-2 : K-dimension ----------------------------------- */
        for (int k = 0; k < p; k += PC)
        {
            int kc = std::min(PC, p - k);

            /* ---- B 패널 : kc × mc  (row-major, kc fast) ------------- */
            for (int jr = 0; jr < mc; jr += MR)
            {
                int mr = std::min(MR, mc - jr);
                for (int l = 0; l < kc; ++l)
                {
                    const double* Bsrc = B + (k + l)*ldb + (j + jr);
                    double*       Bdst = Bpanel + jr*kc + l*MR;  /* l fast */
                    for (int jj = 0; jj <mr; ++jj)  Bdst[jj] = Bsrc[jj];
                    for (int jj = mr; jj < MR; ++jj) Bdst[jj] = 0.0;
                }
            }

            /* ---- Loop-1 : M-dimension (A,C 행) ---------------------- */
            for (int i = 0; i < n; i += NC)
            {
                int nc = std::min(NC, n - i);

                /* A 패널 : nc × kc (col-major, NR fast) */
                for (int ir = 0; ir < nc; ir += NR)
                {
                    int nr = std::min(NR, nc - ir);
                    for (int l = 0; l < kc; ++l)
                    {
                        const double* Asrc = A + (i + ir)*lda + (k + l);
                        double*       Adst = Apanel + ir*kc + l*NR; /* l fast */
                        for (int ii = 0; ii < nr; ++ii)
                            Adst[ii] = Asrc[ii*lda];      /* row-major read */
                        for (int ii = nr; ii < NR; ++ii)  /* padding */
                            Adst[ii] = 0.0;
                    }
                }

                /* ---- 마이크로-커널 호출 ------------------------------ */
                double* Cpanel = C + (i)*ldc + j;         /* C(i,j) 블록 */
                for (int ir = 0; ir < nc; ir += NR)
                {
                    double* Crow = Cpanel + ir*ldc;       /* C 행 시작 */
                    for (int jr = 0; jr < mc; jr += MR)
                    {
                        double* Cblk = Crow + jr;         /* 4×8 타일 */
                        neon_kernel_4x8(kc,
                                        Apanel + ir*kc, NR,   /* A 패널 */
                                        Bpanel + jr*kc, MR,   /* B 패널 */
                                        Cblk, ldc);           /* C */
                    }
                }
            } /* end i-loop */
        }     /* end k-loop */
    }         /* end j-loop */
}

void matmul_naive(const int n, const int p, const int m, const double* A, const int lda,
                  const double* B, const int ldb, double* C, const int ldc) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i * ldc + j] = 0.0;
            for (int k = 0; k < p; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

bool verify_results(const int n, const int p, const int m, const double* A, const int lda,
                    const double* B, const int ldb, double* C, const int ldc) {
    std::vector<double> C_naive(n * m, 0.0);
    std::vector<double> C_opt(n * m, 0.0);
    
    matmul_naive(n, p, m, A, lda, B, ldb, C_naive.data(), ldc);
    matmul_neon_kernel_launcher(n, p, m, A, lda, B, ldb, C_opt.data(), ldc);
    
    const double epsilon = 1e-10;
    bool match = true;

    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < m; ++j) {
            double diff = std::abs(C_naive[i * ldc + j] - C_opt[i * ldc + j]);
            if (diff > epsilon) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                            << "Naive: " << C_naive[i * ldc + j] << ", "
                            << "Optimized: " << C_opt[i * ldc + j] << std::endl;
                    match = false;
            }
        }
    }
    return match;
}

void generate_random_matrix(int rows, int cols, std::vector<double>& matrix) {
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dist(mt);
        }
    }
}

template <typename Func>
double benchmark(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char* argv[]) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 15);
    auto perform_checks = true;
    for (auto i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-checks") {
            std::cout << "Skipping verification checks." << std::endl;
            perform_checks = false;
        }
    }
    std::ofstream benchmark_file("benchmark_results.csv");
    if (!benchmark_file.is_open()) {
        std::cerr << "Error opening benchmark_results.csv for writing." << std::endl;
        return 1;
    }

    // Write header to the CSV file
    benchmark_file << "n, p ,m ,GFLOPS1,GFLOPS2,GFLOPS3,GFLOPS4,GFLOPS5,Verified" << std::endl;

    std::vector<int>sizes;

    for (auto size = 20; size <= 4096; size += 20) {
        sizes.push_back(size);
    }

    // for (auto size = 128; size <= 4096; size += 8) {
    //     sizes.push_back(size);
    // }

    const int num_trials = 5;

    for (auto size : sizes) {

        std::cout << "Running benchmarks for size: " << size << "..." << std::endl;
        int n = size, p = size, m = size;
        auto flop_count = static_cast<double>(n) * p * m * 2.0;

        std::vector<double> A(n * p);
        std::vector<double> B(p * m);
        std::vector<double> C(n * m, 0.0);

        generate_random_matrix(n, p, A);
        generate_random_matrix(p, m, B);

        bool verified = true;
        if (perform_checks) {
            verified = verify_results(n, p, m, A.data(), p, B.data(), m, C.data(), m);
            if (!verified) {
                std::cout << "Verification failed for size: " << size << std::endl;
                exit(1);
            }
        }

        benchmark_file << n << "," << p << "," << m;

        for (auto trial = 0; trial < num_trials; ++trial) {
            std::vector<double> C_test = C;
            auto elapsed_time = benchmark([&]() {
                // Call the matrix multiplication function here
                matmul_neon_kernel_launcher(n, p, m, A.data(), p, B.data(), m, C_test.data(), m);
            });
            auto flops = flop_count / elapsed_time * 1e-9; // Convert to GFLOPS
            benchmark_file << "," << flops;
        }

        if (perform_checks) {
            benchmark_file << ',' << (verified ? "True" : "False");
        }

        benchmark_file << std::endl;
    }

    benchmark_file.close();
    std::cout << "Benchmarking completed. Results saved to benchmark_results.csv." << std::endl;
    
    return 0;  
}
