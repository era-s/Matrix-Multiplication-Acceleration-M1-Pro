#include "m1_matmul/dgemm.hpp"

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "This implementation requires AArch64 Advanced SIMD (NEON)."
#endif

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace m1_matmul {
namespace {

constexpr int kKernelRows = 10;
constexpr int kKernelCols = 4;
constexpr int kDepthBlock = 640;
constexpr int kRowBlock = 520;
constexpr int kColBlock = 768;

void validate_arguments(const double* a, const double* b, const double* c,
                        int rows, int inner, int cols) {
    if (rows < 0 || inner < 0 || cols < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    if (rows > 0 && cols > 0 && c == nullptr) {
        throw std::invalid_argument("C must not be null");
    }
    if (rows > 0 && inner > 0 && a == nullptr) {
        throw std::invalid_argument("A must not be null");
    }
    if (inner > 0 && cols > 0 && b == nullptr) {
        throw std::invalid_argument("B must not be null");
    }
}

void pack_a_panel(const double* a, int lda, int row, int depth,
                  int row_count, int depth_count, double* packed) {
    for (int row_offset = 0; row_offset < row_count;
         row_offset += kKernelRows) {
        const int tile_rows =
            std::min(kKernelRows, row_count - row_offset);
        for (int k = 0; k < depth_count; ++k) {
            const double* source =
                a + (row + row_offset) * lda + depth + k;
            double* destination =
                packed + row_offset * depth_count + k * kKernelRows;

            int tile_row = 0;
            for (; tile_row < tile_rows; ++tile_row) {
                destination[tile_row] = source[tile_row * lda];
            }
            for (; tile_row < kKernelRows; ++tile_row) {
                destination[tile_row] = 0.0;
            }
        }
    }
}

void pack_b_panel(const double* b, int ldb, int depth, int col,
                  int depth_count, int col_count, double* packed) {
    for (int col_offset = 0; col_offset < col_count;
         col_offset += kKernelCols) {
        const int tile_cols =
            std::min(kKernelCols, col_count - col_offset);
        for (int k = 0; k < depth_count; ++k) {
            const double* source =
                b + (depth + k) * ldb + col + col_offset;
            double* destination =
                packed + col_offset * depth_count + k * kKernelCols;

            int tile_col = 0;
            for (; tile_col < tile_cols; ++tile_col) {
                destination[tile_col] = source[tile_col];
            }
            for (; tile_col < kKernelCols; ++tile_col) {
                destination[tile_col] = 0.0;
            }
        }
    }
}

inline void neon_kernel_10x4(int depth_count, const double* __restrict a,
                             const double* __restrict b,
                             double* __restrict c, int ldc) {
    float64x2_t c00 = vld1q_f64(c + 0 * ldc + 0);
    float64x2_t c01 = vld1q_f64(c + 0 * ldc + 2);
    float64x2_t c10 = vld1q_f64(c + 1 * ldc + 0);
    float64x2_t c11 = vld1q_f64(c + 1 * ldc + 2);
    float64x2_t c20 = vld1q_f64(c + 2 * ldc + 0);
    float64x2_t c21 = vld1q_f64(c + 2 * ldc + 2);
    float64x2_t c30 = vld1q_f64(c + 3 * ldc + 0);
    float64x2_t c31 = vld1q_f64(c + 3 * ldc + 2);
    float64x2_t c40 = vld1q_f64(c + 4 * ldc + 0);
    float64x2_t c41 = vld1q_f64(c + 4 * ldc + 2);
    float64x2_t c50 = vld1q_f64(c + 5 * ldc + 0);
    float64x2_t c51 = vld1q_f64(c + 5 * ldc + 2);
    float64x2_t c60 = vld1q_f64(c + 6 * ldc + 0);
    float64x2_t c61 = vld1q_f64(c + 6 * ldc + 2);
    float64x2_t c70 = vld1q_f64(c + 7 * ldc + 0);
    float64x2_t c71 = vld1q_f64(c + 7 * ldc + 2);
    float64x2_t c80 = vld1q_f64(c + 8 * ldc + 0);
    float64x2_t c81 = vld1q_f64(c + 8 * ldc + 2);
    float64x2_t c90 = vld1q_f64(c + 9 * ldc + 0);
    float64x2_t c91 = vld1q_f64(c + 9 * ldc + 2);

    for (int k = 0; k < depth_count; ++k) {
        const double* packed_a = a + k * kKernelRows;
        const double* packed_b = b + k * kKernelCols;

        const float64x2_t a0 = vld1q_dup_f64(packed_a + 0);
        const float64x2_t a1 = vld1q_dup_f64(packed_a + 1);
        const float64x2_t a2 = vld1q_dup_f64(packed_a + 2);
        const float64x2_t a3 = vld1q_dup_f64(packed_a + 3);
        const float64x2_t a4 = vld1q_dup_f64(packed_a + 4);
        const float64x2_t a5 = vld1q_dup_f64(packed_a + 5);
        const float64x2_t a6 = vld1q_dup_f64(packed_a + 6);
        const float64x2_t a7 = vld1q_dup_f64(packed_a + 7);
        const float64x2_t a8 = vld1q_dup_f64(packed_a + 8);
        const float64x2_t a9 = vld1q_dup_f64(packed_a + 9);
        const float64x2_t b0 = vld1q_f64(packed_b + 0);
        const float64x2_t b1 = vld1q_f64(packed_b + 2);

        c00 = vfmaq_f64(c00, a0, b0);
        c01 = vfmaq_f64(c01, a0, b1);
        c10 = vfmaq_f64(c10, a1, b0);
        c11 = vfmaq_f64(c11, a1, b1);
        c20 = vfmaq_f64(c20, a2, b0);
        c21 = vfmaq_f64(c21, a2, b1);
        c30 = vfmaq_f64(c30, a3, b0);
        c31 = vfmaq_f64(c31, a3, b1);
        c40 = vfmaq_f64(c40, a4, b0);
        c41 = vfmaq_f64(c41, a4, b1);
        c50 = vfmaq_f64(c50, a5, b0);
        c51 = vfmaq_f64(c51, a5, b1);
        c60 = vfmaq_f64(c60, a6, b0);
        c61 = vfmaq_f64(c61, a6, b1);
        c70 = vfmaq_f64(c70, a7, b0);
        c71 = vfmaq_f64(c71, a7, b1);
        c80 = vfmaq_f64(c80, a8, b0);
        c81 = vfmaq_f64(c81, a8, b1);
        c90 = vfmaq_f64(c90, a9, b0);
        c91 = vfmaq_f64(c91, a9, b1);
    }

    vst1q_f64(c + 0 * ldc + 0, c00);
    vst1q_f64(c + 0 * ldc + 2, c01);
    vst1q_f64(c + 1 * ldc + 0, c10);
    vst1q_f64(c + 1 * ldc + 2, c11);
    vst1q_f64(c + 2 * ldc + 0, c20);
    vst1q_f64(c + 2 * ldc + 2, c21);
    vst1q_f64(c + 3 * ldc + 0, c30);
    vst1q_f64(c + 3 * ldc + 2, c31);
    vst1q_f64(c + 4 * ldc + 0, c40);
    vst1q_f64(c + 4 * ldc + 2, c41);
    vst1q_f64(c + 5 * ldc + 0, c50);
    vst1q_f64(c + 5 * ldc + 2, c51);
    vst1q_f64(c + 6 * ldc + 0, c60);
    vst1q_f64(c + 6 * ldc + 2, c61);
    vst1q_f64(c + 7 * ldc + 0, c70);
    vst1q_f64(c + 7 * ldc + 2, c71);
    vst1q_f64(c + 8 * ldc + 0, c80);
    vst1q_f64(c + 8 * ldc + 2, c81);
    vst1q_f64(c + 9 * ldc + 0, c90);
    vst1q_f64(c + 9 * ldc + 2, c91);
}

void accumulate_edge_tile(const double* a, const double* b, double* c,
                          int lda, int ldb, int ldc, int row_begin,
                          int row_end, int col_begin, int col_end,
                          int depth_begin, int depth_count) {
    for (int row = row_begin; row < row_end; ++row) {
        for (int col = col_begin; col < col_end; ++col) {
            double value = c[row * ldc + col];
            for (int k = 0; k < depth_count; ++k) {
                value += a[row * lda + depth_begin + k] *
                         b[(depth_begin + k) * ldb + col];
            }
            c[row * ldc + col] = value;
        }
    }
}

}  // namespace

void dgemm_naive(const double* a, const double* b, double* c,
                 int rows, int inner, int cols) {
    validate_arguments(a, b, c, rows, inner, cols);
    if (rows == 0 || cols == 0) {
        return;
    }

    std::fill(c, c + static_cast<std::size_t>(rows) * cols, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double value = 0.0;
            for (int k = 0; k < inner; ++k) {
                value += a[row * inner + k] * b[k * cols + col];
            }
            c[row * cols + col] = value;
        }
    }
}

void dgemm_neon(const double* a, const double* b, double* c,
                int rows, int inner, int cols) {
    validate_arguments(a, b, c, rows, inner, cols);
    if (rows == 0 || cols == 0) {
        return;
    }

    std::fill(c, c + static_cast<std::size_t>(rows) * cols, 0.0);
    if (inner == 0) {
        return;
    }

    thread_local std::vector<double> a_panel(
        static_cast<std::size_t>(kRowBlock) * kDepthBlock);
    thread_local std::vector<double> b_panel(
        static_cast<std::size_t>(kColBlock) * kDepthBlock);

    // j-k-i order: reuse a packed B panel across row blocks, then pack A
    // close to the point where each micro-kernel consumes it.
    for (int col = 0; col < cols; col += kColBlock) {
        const int col_count = std::min(kColBlock, cols - col);
        const int full_cols = col_count - col_count % kKernelCols;

        for (int depth = 0; depth < inner; depth += kDepthBlock) {
            const int depth_count =
                std::min(kDepthBlock, inner - depth);
            pack_b_panel(b, cols, depth, col, depth_count, col_count,
                         b_panel.data());

            for (int row = 0; row < rows; row += kRowBlock) {
                const int row_count = std::min(kRowBlock, rows - row);
                const int full_rows = row_count - row_count % kKernelRows;
                pack_a_panel(a, inner, row, depth, row_count, depth_count,
                             a_panel.data());

                for (int row_offset = 0; row_offset < full_rows;
                     row_offset += kKernelRows) {
                    for (int col_offset = 0; col_offset < full_cols;
                         col_offset += kKernelCols) {
                        neon_kernel_10x4(
                            depth_count,
                            a_panel.data() + row_offset * depth_count,
                            b_panel.data() + col_offset * depth_count,
                            c + (row + row_offset) * cols + col + col_offset,
                            cols);
                    }
                }

                // The tuned kernel only handles complete 10x4 tiles. The two
                // non-overlapping scalar regions make the public API safe for
                // arbitrary matrix shapes without slowing down full tiles.
                accumulate_edge_tile(
                    a, b, c, inner, cols, cols, row + full_rows,
                    row + row_count, col, col + col_count, depth, depth_count);
                accumulate_edge_tile(
                    a, b, c, inner, cols, cols, row, row + full_rows,
                    col + full_cols, col + col_count, depth, depth_count);
            }
        }
    }
}

}  // namespace m1_matmul
