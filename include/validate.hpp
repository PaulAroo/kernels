#pragma once

#include <istream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip> // For std::setprecision
#include <limits>    // For std::numeric_limits

template<typename T>
void compareSequentialAndParallelResults(
  const std::vector<T>& parr, const std::vector<T>& seq
) {
    // Relative tolerance: Check if the difference is more than 0.01% of the larger value.
    const T relative_tolerance = static_cast<T>(1e-4);
    
    // Absolute tolerance: For comparisons near zero.
    const T absolute_tolerance = std::numeric_limits<T>::epsilon() * 100;

    for(size_t i = 0; i < parr.size(); ++i) {
        T diff = std::abs(parr[i] - seq[i]);

        // The check: is the difference larger than a tolerance that scales with the values?
        // Or, if values are very close to zero, is it larger than a small absolute tolerance?
        if (diff > std::max(absolute_tolerance, relative_tolerance * std::max(std::abs(parr[i]), std::abs(seq[i])))) {
            std::cout << "Error: results do not match at index " << i << ".\n"
                      << "  GPU Result: " << std::fixed << std::setprecision(15) << parr[i] << "\n"
                      << "  CPU Result: " << std::fixed << std::setprecision(15) << seq[i] << "\n"
                      << "  Difference: " << diff << std::endl;
            exit(1);
        }
    }
}

// Specialization for integers, where exact comparison is correct
template<>
void compareSequentialAndParallelResults<int>(const std::vector<int>& parr, const std::vector<int>& seq) {
    for(size_t i = 0; i < parr.size(); ++i) {
        if(parr[i] != seq[i]) {
            std::cout << "Error: results do not match at index " << i << ", " 
                      << parr[i] << " is not equal to " << seq[i] << "\n";
            exit(1);
        }
    }
}