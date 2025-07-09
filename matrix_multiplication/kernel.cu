/*
 * Matrix Multiplication

 * - A : M * K
 * - B : K * N
 * - C : M * N
 *
 * All matrices are stored in row-major format
 */

#include <iostream>
#include <cassert>
#include "utils.hpp"
#include "validate.hpp"

#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */
#define NUM_THREADS_PER_DIM 32

 /*
 * Each thread computes a pofloat in output matrix C
 * Memory access -> Global memory
 */
template<typename T>
__global__ void naive_kernel(
  T const* const A, size_t const ldA,
  T const* const B, size_t const ldB,
  T* const C, size_t const ldC,
  size_t const M, size_t const N, size_t const K
) {
  size_t const row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t const col = blockDim.x * blockIdx.x + threadIdx.x;

  if(col < N && row < M ) {
    T accumulator = static_cast<T>(0);
    for(size_t i = 0; i < K; ++i) {
      accumulator += A[row * ldA + i] * B[i * ldB + col];
    }
    C[row * ldC + col] = accumulator;
  }
}

template<typename T>
void computeSequentialMatrixMul(std::vector<T>& A, std::vector<T>& B, std::vector<T>& C,
  size_t const K, size_t const N
) {
  for(size_t i = 0; i < C.size(); ++i) {
    size_t const row = i / N;
    size_t const col = i % N;
    T accumulator = static_cast<T>(0);
    for(size_t k = 0; k < K; ++k) {
      accumulator += A[row * K + k] * B[k * N + col];
    }
    C[i] = accumulator;
  }
}

int main(int argc, char** argv) {
  if ( argc != 4 ) {
    std::cout << "Usage: " << argv[0] << " <M> <K> <N>" << "\n"
      << "\n"
      << "Computes the product of random matrices (constant seed):\n"
      << "  C = A * B" << "\n"
      << "  where" << "\n"
      << "    A : M x K" << "\n"
      << "    B : K x N" << "\n"
      << "    C : M x N" << "\n"
      << std::endl;
      exit(EXIT_FAILURE);
  }

  size_t const M = std::stoll(argv[1]);
  size_t const K = std::stoll(argv[2]);
  size_t const N = std::stoll(argv[3]);
  size_t const ldA = K;
  size_t const ldB = N;
  size_t const ldC = N;

  // initialize data
  std::vector<float> A(M*K);
  fill_random(A.data(), A.size(), std::seed_seq{0});

  std::vector<float> B(K*N);
  fill_random(B.data(), B.size(), std::seed_seq{1});

  std::vector<float> C(M*N);

  float* A_d;
  float* B_d;
  float* C_d;

  try_CUDA(cudaMalloc((void **) &A_d, sizeof(float) * A.size()));
  try_CUDA(cudaMalloc((void **) &B_d, sizeof(float) * B.size()));
  try_CUDA(cudaMalloc((void **) &C_d, sizeof(float) * C.size()));

  // copy data to global memory
  try_CUDA(cudaMemcpy(A_d, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  try_CUDA(cudaMemcpy(B_d, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 numBlocks(
    (N + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    (M + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    1
  );
  dim3 const numThreads(NUM_THREADS_PER_DIM, NUM_THREADS_PER_DIM, 1);

  long gpu_time_ms = benchmark([&]() {
    naive_kernel<<<numBlocks, numThreads>>>(A_d, ldA, B_d, ldB, C_d, ldC, M, N, K);
    try_CUDA(cudaGetLastError());
    try_CUDA(cudaDeviceSynchronize());
  });

  std::cout << "GPU: " << gpu_time_ms << " ms" << std::endl;

  try_CUDA(cudaMemcpy(C.data(), C_d, sizeof(float) * C.size(), cudaMemcpyDeviceToHost));

  std::vector<float> C_seq(M*N);

  long cpu_time_ms = benchmark([&]() {
    computeSequentialMatrixMul(A, B, C_seq, K, N);
  });
  std::cout << "CPU: " << cpu_time_ms << " ms" << std::endl;

  compareSequentialAndParallelResults(C, C_seq);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return EXIT_SUCCESS;
}