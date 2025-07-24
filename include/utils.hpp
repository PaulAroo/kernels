#pragma once

#include "timer.hpp"
#include <utility>

#include <random>
#include <type_traits>
#include <algorithm>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define try_CUDA(call)                                                      \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d in %s(): %s\n",            \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

template <typename F, typename... Args>
double benchmark(F &&func, int warmup_runs = 10, int actual_runs = 30,
                 Args &&...args) {

  // warm up
  for (size_t i = 0; i < warmup_runs; ++i) {
    func(std::forward<Args>(args)...);
  }
  auto timer = StopWatch<chrono_alias::ms>();
  for (size_t i = 0; i < actual_runs; ++i) {
    timer.start();
    func(std::forward<Args>(args)...);
    timer.stop();
  }
  return timer.getAverageTime().count();
}

/*
 * Dynamically chooses between uniform and real distribution
 * Fixed range {0, 100}
 */
template<typename T>
void fill_random(T* v, size_t const n, std::seed_seq& s) {
  std::mt19937 mersenne_generator{s};

  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<T> distribution{0, 100};
    std::generate_n(v, n, [&]() {
      return distribution(mersenne_generator);
    });
  } else if constexpr (std::is_floating_point<T>::value) {
    std::uniform_real_distribution<T> distribution{0.0, 100.0};
    std::generate_n(v, n, [&]() {
      return distribution(mersenne_generator);
    });
  } else {
    static_assert(std::is_arithmetic<T>::value, "fill_random only supports numeric types.");
  }
}

// Rvalue overload
template<typename T>
inline void fill_random(T* v, size_t const n, std::seed_seq&& s) {
  fill_random<T>(v, n, s);
}


// template<class F>
// long benchmark_one_ms(F&& f) {
  // auto start = std::chrono::steady_clock::now();
  // f();
  // auto end = std::chrono::steady_clock::now();
  // return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
// }

void printGPUInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return;
  }

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device #" << device << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max Grid Size: [" 
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "  Max Threads Dim: ["
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;
  }
}
