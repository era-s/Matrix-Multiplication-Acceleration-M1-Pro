#include "m1_matmul/dgemm.hpp"

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int min_size = 128;
    int max_size = 2048;
    int step = 128;
    int trials = 5;
    std::string output = "results/benchmark-latest.csv";
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --min N       smallest square matrix (default: 128)\n"
        << "  --max N       largest square matrix (default: 2048)\n"
        << "  --step N      size increment (default: 128)\n"
        << "  --trials N    timed trials per size (default: 5)\n"
        << "  --output PATH CSV output (default: results/benchmark-latest.csv)\n"
        << "  --help        show this message\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--min") {
            options.min_size = std::stoi(value);
        } else if (argument == "--max") {
            options.max_size = std::stoi(value);
        } else if (argument == "--step") {
            options.step = std::stoi(value);
        } else if (argument == "--trials") {
            options.trials = std::stoi(value);
        } else if (argument == "--output") {
            options.output = value;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.min_size <= 0 || options.max_size < options.min_size ||
        options.step <= 0 || options.trials <= 0) {
        throw std::invalid_argument("sizes, step, and trials must be positive");
    }
    return options;
}

void fill_random(std::vector<double>& matrix, std::mt19937& generator) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (double& value : matrix) {
        value = distribution(generator);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

        std::ofstream output(options.output);
        if (!output) {
            throw std::runtime_error("could not open " + options.output);
        }
        output << "size,trial,seconds,gflops\n";

        std::mt19937 generator(20250720);
        std::cout << std::fixed << std::setprecision(4);

        for (int size = options.min_size; size <= options.max_size;
             size += options.step) {
            const std::size_t element_count =
                static_cast<std::size_t>(size) * size;
            std::vector<double> a(element_count);
            std::vector<double> b(element_count);
            std::vector<double> c(element_count);
            fill_random(a, generator);
            fill_random(b, generator);

            // Populate thread-local packed buffers and warm the code path.
            m1_matmul::dgemm_neon(a.data(), b.data(), c.data(), size, size,
                                  size);

            std::vector<double> measurements;
            measurements.reserve(options.trials);
            for (int trial = 1; trial <= options.trials; ++trial) {
                const auto begin = std::chrono::steady_clock::now();
                m1_matmul::dgemm_neon(a.data(), b.data(), c.data(), size, size,
                                      size);
                const auto end = std::chrono::steady_clock::now();
                const double seconds =
                    std::chrono::duration<double>(end - begin).count();
                const double operations =
                    2.0 * static_cast<double>(size) * size * size;
                const double gflops = operations / seconds * 1e-9;
                measurements.push_back(gflops);
                output << size << ',' << trial << ',' << std::setprecision(9)
                       << seconds << ',' << std::setprecision(6) << gflops
                       << '\n';
            }

            const double average =
                std::accumulate(measurements.begin(), measurements.end(), 0.0) /
                measurements.size();
            const double peak =
                *std::max_element(measurements.begin(), measurements.end());
            std::cout << "N=" << std::setw(4) << size << "  average "
                      << std::setw(8) << average << " GFLOPS  peak "
                      << std::setw(8) << peak << " GFLOPS\n";
        }

        std::cout << "Saved raw measurements to " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
