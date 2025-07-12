#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>


void rank_one_kernel_4x4(const int k, 
                          const double* A, const int lda,
                          const double* B, const int ldb,
                          double* C, const int ldc) {

    double c00 = C[0 * ldc + 0];
    double c01 = C[0 * ldc + 1];
    double c02 = C[0 * ldc + 2];
    double c03 = C[0 * ldc + 3];

    double c10 = C[1 * ldc + 0];
    double c11 = C[1 * ldc + 1];
    double c12 = C[1 * ldc + 2];
    double c13 = C[1 * ldc + 3];

    double c20 = C[2 * ldc + 0];
    double c21 = C[2 * ldc + 1];
    double c22 = C[2 * ldc + 2];
    double c23 = C[2 * ldc + 3];

    double c30 = C[3 * ldc + 0];
    double c31 = C[3 * ldc + 1];
    double c32 = C[3 * ldc + 2];
    double c33 = C[3 * ldc + 3];

    for (auto l = 0; l < k; l++) {
        double a0 = A[0 * lda + l];
        double a1 = A[1 * lda + l];
        double a2 = A[2 * lda + l];
        double a3 = A[3 * lda + l];

        double b0 = B[l * ldb + 0];
        double b1 = B[l * ldb + 1];
        double b2 = B[l * ldb + 2];
        double b3 = B[l * ldb + 3];

        c00 += a0 * b0;   c01 += a0 * b1;
        c02 += a0 * b2;   c03 += a0 * b3;

        c10 += a1 * b0;   c11 += a1 * b1;
        c12 += a1 * b2;   c13 += a1 * b3;

        c20 += a2 * b0;   c21 += a2 * b1;
        c22 += a2 * b2;   c23 += a2 * b3;

        c30 += a3 * b0;   c31 += a3 * b1;
        c32 += a3 * b2;   c33 += a3 * b3;
    }

    C[0 * ldc + 0] = c00;   C[0 * ldc + 1] = c01;
    C[0 * ldc + 2] = c02;   C[0 * ldc + 3] = c03;

    C[1 * ldc + 0] = c10;   C[1 * ldc + 1] = c11;
    C[1 * ldc + 2] = c12;   C[1 * ldc + 3] = c13;

    C[2 * ldc + 0] = c20;   C[2 * ldc + 1] = c21;
    C[2 * ldc + 2] = c22;   C[2 * ldc + 3] = c23;

    C[3 * ldc + 0] = c30;   C[3 * ldc + 1] = c31;
    C[3 * ldc + 2] = c32;   C[3 * ldc + 3] = c33;
    
}

void matmul_kernel_launcher(const int n, const int p, const int m,
                            const double* A, const int lda,
                            const double* B, const int ldb,
                            double* C, const int ldc) {

    if (n % 4 != 0 || p % 4 != 0 || m % 4 != 0) {
        std::cerr << "Matrix dimensions must be multiples of 4 for this kernel." << std::endl;
        return;
    }

    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < m; j += 4) {
            rank_one_kernel_4x4(p, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
        }
    }
    
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
    matmul_kernel_launcher(n, p, m, A, lda, B, ldb, C_opt.data(), ldc);
    
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
    auto perform_checks = true;
    for (auto i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-checks") {
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

    for (auto size = 4; size <= 124; size += 4) {
        sizes.push_back(size);
    }

    for (auto size = 128; size <= 4096; size += 8) {
        sizes.push_back(size);
    }

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
                matmul_kernel_launcher(n, p, m, A.data(), p, B.data(), m, C_test.data(), m);
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