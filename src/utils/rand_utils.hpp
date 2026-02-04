#pragma once

#include <string>
#include <random>

namespace random_generator {

    inline std::string rand_alphanum(size_t length) {
        static const char alphanum[] = "0123456789"
                                       "abcdefghijklmnopqrstuvwxyz";

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

        std::string result;
        result.reserve(length);

        for(size_t i = 0; i < length; ++i) {
            result += alphanum[dis(gen)];
        }

        return result;
    }

    inline int random_int(int min, int max) {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator);
    }

    /**
     * Returns power-of-2 bits needed to hold current_elements with 1% false positive rate
     * NOTE: This function is not in use right now
     */
    inline size_t calculateOptimalBloomBits(size_t current_elements) {
        if(current_elements == 0) {
            return 1;  // Minimum 1 bit
        }
        // Bloom filter formula for 1% false positive rate
        // m = -n*ln(p) / (ln(2)^2) where p=0.01
        // m ≈ 9.576 * n
        double optimal_bits = 9.576 * static_cast<double>(current_elements);
        size_t bits_needed = static_cast<size_t>(optimal_bits) + 1;  // Add 1 for rounding up

        // Round up to nearest power of 2 for fast bitwise operations
        size_t power_of_2_bits = 0;
        size_t temp = bits_needed - 1;
        while(temp > 0) {
            temp >>= 1;
            power_of_2_bits++;
        }
        return power_of_2_bits;
    }
}  //namespace random_generator