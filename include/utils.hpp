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
double benchmark(F &&func, int warmup_runs = 10, int actual_runs = 50,
                 Args &&...args) {

  // warm up
  for (size_t i = 0; i < warmup_runs; ++i) {
    func(std::forward<Args>(args)...);
  }
  auto timer = StopWatch<chrono_alias::ns>();
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


template<typename T>
void compareSequentialAndParallelResults(std::vector<T> parr, std::vector<T> seq) {
  for(size_t i = 0; i < parr.size(); ++i) {
    if(parr[i] != seq[i]) {
      std::cout << "Error: results do not match at index " << i << ", " << parr[i] << " is not equal to " << seq[i] << "\n";
      exit(1);
    }
  }
}