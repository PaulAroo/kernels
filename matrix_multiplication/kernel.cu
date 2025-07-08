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
// #include <chrono>
#include "utils.hpp"

#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */
#define NUM_THREADS_PER_DIM 32

 /*
 * Each thread computes a point in output matrix C
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

template<class F>
long benchmark_one_ms(F&& f) {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}


template<typename T>
void computeSequential(std::vector<T>& A, std::vector<T>& B, std::vector<T>& C,
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

template<typename T>
void compareSequentialAndParallelResults(std::vector<T> parr, std::vector<T> seq) {
  for(size_t i = 0; i < parr.size(); ++i) {
    if(parr[i] != seq[i]) {
      std::cout << "Error: results do not match at index " << i << ", " << parr[i] << " is not equal to " << seq[i] << "\n";
      exit(1);
    }
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
  std::seed_seq seed_seq{0};

  std::vector<int> A(M*K);
  fill_random(A.data(), A.size(), seed_seq);

  std::vector<int> B(K*N);
  fill_random(B.data(), B.size(), seed_seq);

  std::vector<int> C(M*N);

  int* A_d;
  int* B_d;
  int* C_d;

  cudaMalloc((void **) &A_d, sizeof(int) * A.size());
  cudaMalloc((void **) &B_d, sizeof(int) * B.size());
  cudaMalloc((void **) &C_d, sizeof(int) * C.size());

  // copy data to global memory
  cudaMemcpy(A_d, A.data(), A.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B.data(), B.size() * sizeof(int), cudaMemcpyHostToDevice);

  const dim3 numBlocks(
    (N + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    (M + NUM_THREADS_PER_DIM - 1) / NUM_THREADS_PER_DIM,
    1
  );
  dim3 const numThreads(NUM_THREADS_PER_DIM, NUM_THREADS_PER_DIM, 1);

  long gpu_time_ns = benchmark([&]() {
    naive_kernel<<<numBlocks, numThreads>>>(A_d, ldA, B_d, ldB, C_d, ldC, M, N, K);
    cudaDeviceSynchronize();
  });
  std::cout << "GPU: " << gpu_time_ns << " ns" << std::endl;

  cudaMemcpy(C.data(), C_d, sizeof(int) * C.size(), cudaMemcpyDeviceToHost);

  std::vector<int> C_seq(M*N);

  long cpu_time_ns = benchmark([&]() {
    computeSequential(A, B, C_seq, K, N);
  });
  std::cout << "CPU: " << cpu_time_ns << " ns" << std::endl;

  compareSequentialAndParallelResults(C, C_seq);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}