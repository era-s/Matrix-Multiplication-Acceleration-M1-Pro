#include <arm_neon.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

// Define block sizes
#define NR 14
#define MR 2

// #define PC 280 // 640 ~ 1320 ~1400 아니 왜 l1 보다 커져도 빠르지?
// #define NC 1400
// #define MC 560

// #define PC 280 // 640 ~ 1320 ~1400 아니 왜 l1 보다 커져도 빠르지?
// #define NC 2800
// #define MC 140

#define PC 560 // 640 ~ 1320 ~1400 아니 왜 l1 보다 커져도 빠르지?
#define NC 420
#define MC 256

#define CACHELINE 64
#if defined(__GNUC__) || defined(__clang__)
    #define ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
    #define ALIGN(x) __declspec(align(x))
#else
    #define ALIGN(x)
#endif

ALIGN(CACHELINE) static double Apanel[PC * NC] __attribute__((aligned(16384)));
ALIGN(CACHELINE) static double Bpanel[MC * PC] __attribute__((aligned(16384)));

static inline void neon_kernel_14x2(const int kc,
                                   const double* __restrict A, const int ldA,
                                   const double* __restrict B, const int ldB,
                                   double*       __restrict C, const int ldC)
{
    float64x2_t c0 = vld1q_f64(C + 0*ldC + 0);
    float64x2_t c1 = vld1q_f64(C + 1*ldC + 0);
    float64x2_t c2 = vld1q_f64(C + 2*ldC + 0);
    float64x2_t c3 = vld1q_f64(C + 3*ldC + 0);
    float64x2_t c4 = vld1q_f64(C + 4*ldC + 0);
    float64x2_t c5 = vld1q_f64(C + 5*ldC + 0);
    float64x2_t c6 = vld1q_f64(C + 6*ldC + 0);
    float64x2_t c7 = vld1q_f64(C + 7*ldC + 0);
    float64x2_t c8 = vld1q_f64(C + 8*ldC + 0);
    float64x2_t c9 = vld1q_f64(C + 9*ldC + 0);
    float64x2_t c10 = vld1q_f64(C + 10*ldC + 0);
    float64x2_t c11 = vld1q_f64(C + 11*ldC + 0);
    float64x2_t c12 = vld1q_f64(C + 12*ldC + 0);
    float64x2_t c13 = vld1q_f64(C + 13*ldC + 0);

    for (int l = 0; l < kc; ++l)
    {
        float64x2_t a0 = vld1q_dup_f64(A + l*ldA + 0);
        float64x2_t a1 = vld1q_dup_f64(A + l*ldA + 1);
        float64x2_t a2 = vld1q_dup_f64(A + l*ldA + 2);
        float64x2_t a3 = vld1q_dup_f64(A + l*ldA + 3);
        float64x2_t a4 = vld1q_dup_f64(A + l*ldA + 4);
        float64x2_t a5 = vld1q_dup_f64(A + l*ldA + 5);
        float64x2_t a6 = vld1q_dup_f64(A + l*ldA + 6);
        float64x2_t a7 = vld1q_dup_f64(A + l*ldA + 7);
        float64x2_t a8 = vld1q_dup_f64(A + l*ldA + 8);
        float64x2_t a9 = vld1q_dup_f64(A + l*ldA + 9);
        float64x2_t a10 = vld1q_dup_f64(A + l*ldA + 10);
        float64x2_t a11 = vld1q_dup_f64(A + l*ldA + 11);
        float64x2_t a12 = vld1q_dup_f64(A + l*ldA + 12);
        float64x2_t a13 = vld1q_dup_f64(A + l*ldA + 13);

        float64x2_t b0 = vld1q_f64(B + l*ldB + 0);  

        c0 = vfmaq_f64(c0, a0, b0);
        c1 = vfmaq_f64(c1, a1, b0);  
        c2 = vfmaq_f64(c2, a2, b0);  
        c3 = vfmaq_f64(c3, a3, b0);  
        c4 = vfmaq_f64(c4, a4, b0);  
        c5 = vfmaq_f64(c5, a5, b0);  
        c6 = vfmaq_f64(c6, a6, b0);  
        c7 = vfmaq_f64(c7, a7, b0);  
        c8 = vfmaq_f64(c8, a8, b0);  
        c9 = vfmaq_f64(c9, a9, b0);  
        c10 = vfmaq_f64(c10, a10, b0);  
        c11 = vfmaq_f64(c11, a11, b0);  
        c12 = vfmaq_f64(c12, a12, b0);  
        c13 = vfmaq_f64(c13, a13, b0);  

    }

    vst1q_f64(C + 0*ldC + 0, c0);
    vst1q_f64(C + 1*ldC + 0, c1);
    vst1q_f64(C + 2*ldC + 0, c2);
    vst1q_f64(C + 3*ldC + 0, c3);
    vst1q_f64(C + 4*ldC + 0, c4);
    vst1q_f64(C + 5*ldC + 0, c5);
    vst1q_f64(C + 6*ldC + 0, c6);
    vst1q_f64(C + 7*ldC + 0, c7);
    vst1q_f64(C + 8*ldC + 0, c8);
    vst1q_f64(C + 9*ldC + 0, c9);
    vst1q_f64(C + 10*ldC + 0, c10);
    vst1q_f64(C + 11*ldC + 0, c11);
    vst1q_f64(C + 12*ldC + 0, c12);
    vst1q_f64(C + 13*ldC + 0, c13);
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

                // double* Cpanel = C + (i)*ldc + j;         /* C(i,j) 블록 */
                // for (int jr = 0; jr < mc; jr += MR)
                // {
                //     double* Ccol = Cpanel + jr;       /* C 행 시작 */
                //     for (int ir = 0; ir < nc; ir += NR)
                //     {
                //         double* Cblk = Ccol + ir * ldc;         /* 4×8 타일 */
                //         neon_kernel_14x2(kc,
                //                         Apanel + ir*kc, NR,   /* A 패널 */
                //                         Bpanel + jr*kc, MR,   /* B 패널 */
                //                         Cblk, ldc);           /* C */
                //     }
                // }
                
                /* ---- 마이크로-커널 호출 ------------------------------ */
                double* Cpanel = C + (i)*ldc + j;         /* C(i,j) 블록 */
                for (int ir = 0; ir < nc; ir += NR)
                {
                    double* Crow = Cpanel + ir*ldc;       /* C 행 시작 */
                    for (int jr = 0; jr < mc; jr += MR)
                    {
                        double* Cblk = Crow + jr;         /* 4×8 타일 */
                        neon_kernel_14x2(kc,
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

    for (auto size = 28; size <= 4096; size += 28) {
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
