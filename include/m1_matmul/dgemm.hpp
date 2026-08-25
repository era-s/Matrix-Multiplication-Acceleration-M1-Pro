#pragma once

namespace m1_matmul {

// All matrices use row-major storage:
//   A: rows x inner
//   B: inner x cols
//   C: rows x cols
// Both functions overwrite C with A * B.
void dgemm_naive(const double* a, const double* b, double* c,
                 int rows, int inner, int cols);

// Apple Silicon implementation using a 10x4 FP64 NEON micro-kernel and
// cache-sized packed panels. Remainder rows and columns use a scalar path,
// so dimensions do not need to be multiples of the micro-kernel size.
void dgemm_neon(const double* a, const double* b, double* c,
                int rows, int inner, int cols);

}  // namespace m1_matmul
