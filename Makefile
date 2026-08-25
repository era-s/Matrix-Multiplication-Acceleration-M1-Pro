CXX ?= clang++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS ?=

BUILD_DIR := build
LIB_OBJECT := $(BUILD_DIR)/dgemm_neon.o
BENCHMARK := $(BUILD_DIR)/benchmark
TEST := $(BUILD_DIR)/test_dgemm

.PHONY: all test benchmark clean

all: $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJECT): src/dgemm_neon.cpp include/m1_matmul/dgemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BENCHMARK): benchmarks/benchmark.cpp $(LIB_OBJECT) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(TEST): tests/test_dgemm.cpp $(LIB_OBJECT) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

test: $(TEST)
	./$(TEST)

benchmark: $(BENCHMARK)
	./$(BENCHMARK) --output results/benchmark-latest.csv

clean:
	rm -f $(LIB_OBJECT) $(BENCHMARK) $(TEST)
	rmdir $(BUILD_DIR) 2>/dev/null || true
