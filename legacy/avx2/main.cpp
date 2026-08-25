#include <iostream>
#include <iomanip>
#include "gemm.h"
#include <cstring>

void print_matrix(const float* mat, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(3) << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    constexpr int N = 1024, P = 1024, M = 1024;

    float* A = new float[N * P];
    float* B = new float[P * M];
    float* C = new float[N * M];
    memset(C, 0, N*M*sizeof(float));

    // A[i][j] = i + 1
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j)
            A[i * P + j] = static_cast<float>(i + 1);

    // B = Identity matrix
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < M; ++j)
            B[i * M + j] = (i == j) ? 1.0f : 0.0f;


    // gemm_reordered_unroll(C, A, B, N, P, M);
    // gemm_intrinsic(C, A, B, N, M, P);
    gemm_reordered_intrinsic(C, A, B, N, M, P);

    // print_matrix(A, N, P, "A");
    // print_matrix(B, P, M, "B (Identity)");
    // print_matrix(C, N, M, "C = A * B using AVX-512");

    return 0;
}
