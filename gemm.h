#pragma once

void gemm(float* C, float* A, float* B, int N, int M, int P);
void gemm_reordered(float* C, float* A, float* B, int N, int M, int P);
void gemm_unroll(float* C, float* A, float* B, int N, int M, int P);
void gemm_reordered_unroll(float* C, float* A, float* B, int N, int M, int P);