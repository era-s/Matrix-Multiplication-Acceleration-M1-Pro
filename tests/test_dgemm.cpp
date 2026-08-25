#include "m1_matmul/dgemm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace {

bool check_shape(int rows, int inner, int cols, std::mt19937& generator) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> a(static_cast<std::size_t>(rows) * inner);
    std::vector<double> b(static_cast<std::size_t>(inner) * cols);
    std::vector<double> expected(static_cast<std::size_t>(rows) * cols);
    std::vector<double> actual(static_cast<std::size_t>(rows) * cols, 7.0);

    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });

    m1_matmul::dgemm_naive(a.data(), b.data(), expected.data(), rows, inner,
                           cols);
    m1_matmul::dgemm_neon(a.data(), b.data(), actual.data(), rows, inner,
                          cols);

    double max_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = std::abs(expected[index] - actual[index]);
        const double tolerance = 1e-10 * (1.0 + std::abs(expected[index]));
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cerr << "FAIL " << rows << "x" << inner << "x" << cols
                      << " at element " << index << ": expected "
                      << expected[index] << ", got " << actual[index]
                      << ", error " << error << '\n';
            return false;
        }
    }

    std::cout << "PASS " << rows << "x" << inner << "x" << cols
              << " (max error " << max_error << ")\n";
    return true;
}

}  // namespace

int main() {
    std::mt19937 generator(20250720);
    const std::vector<std::tuple<int, int, int>> shapes = {
        {1, 1, 1},       {3, 5, 7},      {10, 17, 4},
        {11, 17, 5},     {23, 33, 19},   {64, 64, 64},
        {521, 17, 13},   {13, 641, 7},   {11, 9, 769},
    };

    for (const auto& [rows, inner, cols] : shapes) {
        if (!check_shape(rows, inner, cols, generator)) {
            return 1;
        }
    }

    std::vector<double> zero_inner(6, 1.0);
    m1_matmul::dgemm_neon(nullptr, nullptr, zero_inner.data(), 2, 0, 3);
    if (!std::all_of(zero_inner.begin(), zero_inner.end(),
                     [](double value) { return value == 0.0; })) {
        std::cerr << "FAIL zero inner dimension did not clear C\n";
        return 1;
    }

    std::cout << "All correctness checks passed.\n";
    return 0;
}
