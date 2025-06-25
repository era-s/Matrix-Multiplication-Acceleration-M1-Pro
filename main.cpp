#include <iostream>
#include <iomanip>
#include "gemm.h"

void print_matrix(const float* mat, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(5) << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    constexpr int N = 4, P = 4, M = 4;

    float A[N * P] = {
        1, 2, 3, 4, 
        5, 6, 7, 8, 
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float B[P * M] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    float C1[N * M] = {};
    float C2[N * M] = {};
    float C3[N * M] = {};
    float C4[N * M] = {};

    gemm(C1, A, B, N, M, P);
    gemm_reordered(C2, A, B, N, M, P);
    gemm_unroll(C3, A, B, N, M, P);
    gemm_reordered_unroll(C4, A, B, N, M, P);

    print_matrix(A, N, P, "A");
    print_matrix(B, P, M, "B (Identity)");
    print_matrix(C1, N, M, "C1 = gemm");
    print_matrix(C2, N, M, "C2 = gemm_reordered");
    print_matrix(C3, N, M, "C3 = gemm_unroll");
    print_matrix(C4, N, M, "C4 = gemm_reordered_unroll");

    return 0;
}
