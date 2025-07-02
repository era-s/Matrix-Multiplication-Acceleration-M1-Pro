#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>


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
    return elapsed.count()
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

        // 검증 코드 추가해야함
        bool verified = false;

        benchmark_file << n << "," << p << "," << m;

        for (auto trial = 0; trial < num_trials; ++trial) {
            std::vector<double> C_test = C;
            auto elapsed_time = benchmark([&]() {
                // Call the matrix multiplication function here
                // For example: matrix_multiply(C_test.data(), n, p, m, A.data(), B.data());
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