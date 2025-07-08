#pragma once

#include <random>
#include <type_traits>
#include <algorithm>

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
