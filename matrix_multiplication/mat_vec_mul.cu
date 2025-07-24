/*
 * Matrix Vector Multiplication

 * - A : M * N    input matrix
 * - B : N        input vector
 * - C : N        output vector

 * Matrix is stored in row-major format

 * - ldA: leading dimension of A === N
 */

#include <iostream>
#include <cassert>
#include "utils.hpp"
#include "validate.hpp"

#define	EXIT_FAILURE	1	/* Fait siling extatus.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */
#define THREADS_PER_BLOCK 32


template<typename T>
__global__ void matVecMul(
  T const* const A, size_t const ldA,
  T const* const B, T* const C,
  size_t const M, size_t const N
) {
  size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < N) {
    T acc = static_cast<T>(0);
    for(size_t i = 0; i < ldA; ++i) {
      acc += A[(tid * ldA) + i] * B[i];
    }
    C[tid] = acc;
  }
}

template<typename T>
void computeSequentialMatVecMul(
  const std::vector<T>& A, size_t const ldA,
  const std::vector<T>& B, std::vector<T>& C
) {

  for(size_t i = 0; i < C.size(); ++i) {
    T acc = static_cast<T>(0);
    for(size_t j = 0; j < ldA; ++j) {
      acc += A[(i * ldA) + j] * B[j];
    }

    C[i] = acc;
  }

}

int main(int argc, char** argv) {
  if ( argc != 3 ) {
    std::cout << "Usage: " << argv[0] << " <M> <N>" << "\n"
      << "\n"
      << "Computes the product of a matrix and a vector (constant seed):\n"
      << "  C = A * B" << "\n"
      << "  where" << "\n"
      << "    A : M x N" << "\n"
      << "    B : N" << "\n"
      << "    C : N" << "\n"
      << "          M -> number of rows, N -> number of columns" << "\n"
      << std::endl;
      exit(EXIT_FAILURE);
  }

  // printGPUInfo();

  size_t const M = std::stoll(argv[1]);
  size_t const N = std::stoll(argv[2]);
  size_t const ldA = N;

  

  // initialize data
  std::vector<float> A(M*N);
  fill_random(A.data(), A.size(), std::seed_seq{0});

  std::vector<float> B(N);
  fill_random(B.data(), B.size(), std::seed_seq{1});

  std::vector<float> C(N);

  float* A_d;
  float* B_d;
  float* C_d;

  try_CUDA(cudaMalloc((void **) &A_d, sizeof(float) * A.size()));
  try_CUDA(cudaMalloc((void **) &B_d, sizeof(float) * B.size()));
  try_CUDA(cudaMalloc((void **) &C_d, sizeof(float) * C.size()));

  // copy data to global memory
  try_CUDA(cudaMemcpy(A_d, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  try_CUDA(cudaMemcpy(B_d, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 numBlocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 const numThreads(THREADS_PER_BLOCK);

  long gpu_time_ms = benchmark([&]() {
    matVecMul<<<numBlocks, numThreads>>>(A_d, ldA, B_d, C_d, M, N);
    try_CUDA(cudaGetLastError());
    try_CUDA(cudaDeviceSynchronize());
  });

  std::cout << "GPU: " << gpu_time_ms << " ms" << std::endl;

  try_CUDA(cudaMemcpy(C.data(), C_d, sizeof(float) * C.size(), cudaMemcpyDeviceToHost));

  std::vector<float> C_seq(N);

  long cpu_time_ms = benchmark([&]() {
    computeSequentialMatVecMul(A, ldA, B, C_seq);
  });
  std::cout << "CPU: " << cpu_time_ms << " ms" << std::endl;

  compareSequentialAndParallelResults(C, C_seq);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return EXIT_SUCCESS;
}