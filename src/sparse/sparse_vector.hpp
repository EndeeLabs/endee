#pragma once

#include <vector>
#include <cstring>
#include <stdexcept>
#include "mdbx/mdbx.h"

namespace ndd {

    struct SparseVector {
        std::vector<uint32_t> indices;  // term IDs (sorted)
        std::vector<float> values;      // corresponding values

        // Default constructor
        SparseVector() = default;

        // Constructor from packed data
        SparseVector(const uint8_t* data, size_t data_size) {
            if(data_size < sizeof(uint16_t)) {
                throw std::runtime_error("Invalid packed data: insufficient size for nnz field");
            }

            const uint8_t* ptr = data;

            // Read number of non-zero elements (uint16_t)
            uint16_t nnz;
            std::memcpy(&nnz, ptr, sizeof(uint16_t));
            ptr += sizeof(uint16_t);

            // Validate remaining data size: nnz * (4 + 2) bytes
            size_t expected_size = sizeof(uint16_t) + (nnz * (sizeof(uint32_t) + sizeof(uint16_t)));
            if(data_size != expected_size) {
                throw std::runtime_error("Invalid packed data: size mismatch");
            }

            if(nnz > 0) {
                indices.resize(nnz);
                values.resize(nnz);

                // Read term IDs (uint32_t each)
                std::memcpy(indices.data(), ptr, nnz * sizeof(uint32_t));
                ptr += nnz * sizeof(uint32_t);

                // Read quantized values (uint16_t each) and convert to float
                std::vector<uint16_t> fp16_values(nnz);
                std::memcpy(fp16_values.data(), ptr, nnz * sizeof(uint16_t));

                // Convert FP16 to float
                for(size_t i = 0; i < nnz; ++i) {
                    values[i] = fp16_to_float(fp16_values[i]);
                }
            }
        }

        // Convenience constructors
        explicit SparseVector(const MDBX_val& mdb_val) :
            SparseVector(static_cast<const uint8_t*>(mdb_val.iov_base), mdb_val.iov_len) {}

        explicit SparseVector(const std::vector<uint8_t>& packed_data) :
            SparseVector(packed_data.data(), packed_data.size()) {}

        // Pack sparse vector into binary format: nnz(u16) + [term_ids(u32)] + [values(f16)]
        std::vector<uint8_t> pack() const {
            if(indices.size() != values.size()) {
                throw std::runtime_error("SparseVector indices and values size mismatch");
            }

            uint16_t nnz = static_cast<uint16_t>(indices.size());

            // Calculate total size: nnz(2) + term_ids(4*nnz) + values(2*nnz)
            size_t total_size =
                    sizeof(uint16_t) + (nnz * sizeof(uint32_t)) + (nnz * sizeof(uint16_t));

            // Single allocation
            std::vector<uint8_t> packed(total_size);
            uint8_t* ptr = packed.data();

            // Write nnz
            std::memcpy(ptr, &nnz, sizeof(uint16_t));
            ptr += sizeof(uint16_t);

            if(nnz > 0) {
                // Write term IDs
                std::memcpy(ptr, indices.data(), nnz * sizeof(uint32_t));
                ptr += nnz * sizeof(uint32_t);

                // Convert float values to FP16 and write
                std::vector<uint16_t> fp16_values(nnz);
                for(size_t i = 0; i < nnz; ++i) {
                    fp16_values[i] = float_to_fp16(values[i]);
                }
                std::memcpy(ptr, fp16_values.data(), nnz * sizeof(uint16_t));
            }

            return packed;
        }

        // Dot product overloads
        float dot(const SparseVector& other) const {
            float result = 0.0f;
            size_t i = 0, j = 0;
            while(i < indices.size() && j < other.indices.size()) {
                if(indices[i] == other.indices[j]) {
                    result += values[i] * other.values[j];
                    ++i;
                    ++j;
                } else if(indices[i] < other.indices[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
            return result;
        }

        // Dot product with packed data (zero-copy)
        float dot(const uint8_t* packed_data, size_t data_size) const {
            if(data_size < sizeof(uint16_t) || indices.empty()) {
                return 0.0f;
            }

            const uint8_t* ptr = packed_data;

            // Read nnz
            uint16_t other_nnz;
            std::memcpy(&other_nnz, ptr, sizeof(uint16_t));
            ptr += sizeof(uint16_t);

            if(other_nnz == 0) {
                return 0.0f;
            }

            // Direct pointer access to packed data
            const uint32_t* other_indices = reinterpret_cast<const uint32_t*>(ptr);
            const uint16_t* other_fp16_values =
                    reinterpret_cast<const uint16_t*>(ptr + other_nnz * sizeof(uint32_t));

            // Two-pointer intersection
            float result = 0.0f;
            size_t this_idx = 0;

            for(uint16_t other_idx = 0; other_idx < other_nnz && this_idx < indices.size();) {
                uint32_t this_index = indices[this_idx];
                uint32_t other_index = other_indices[other_idx];

                if(this_index == other_index) {
                    float other_value = fp16_to_float(other_fp16_values[other_idx]);
                    result += values[this_idx] * other_value;
                    ++this_idx;
                    ++other_idx;
                } else if(this_index < other_index) {
                    ++this_idx;
                } else {
                    ++other_idx;
                }
            }

            return result;
        }

        // Convenience overloads
        float dot(const MDBX_val& mdb_val) const {
            return dot(static_cast<const uint8_t*>(mdb_val.iov_base), mdb_val.iov_len);
        }

        float dot(const std::vector<uint8_t>& packed_data) const {
            return dot(packed_data.data(), packed_data.size());
        }

        // Utility methods
        bool empty() const { return indices.empty(); }
        size_t size() const { return indices.size(); }
        void clear() {
            indices.clear();
            values.clear();
        }

    private:
        // Simple FP16 conversion (you might want to use a proper library)
        static uint16_t float_to_fp16(float f) {
            // Simplified conversion - use proper library in production
            union {
                float f;
                uint32_t i;
            } u = {f};
            uint32_t sign = (u.i >> 31) << 15;
            uint32_t exp = ((u.i >> 23) & 0xff) - 127 + 15;
            uint32_t frac = (u.i >> 13) & 0x3ff;

            if(exp <= 0) {
                return static_cast<uint16_t>(sign);
            }
            if(exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00);
            }

            return static_cast<uint16_t>(sign | (exp << 10) | frac);
        }

        static float fp16_to_float(uint16_t h) {
            // Simplified conversion - use proper library in production
            uint32_t sign = (h >> 15) << 31;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t frac = h & 0x3ff;

            if(exp == 0) {
                if(frac == 0) {
                    union {
                        uint32_t i;
                        float f;
                    } u = {sign};
                    return u.f;
                }
                // Denormalized
                exp = 127 - 15 + 1 - 10;
                while((frac & 0x400) == 0) {
                    frac <<= 1;
                    exp--;
                }
                frac &= 0x3ff;
            } else if(exp == 31) {
                exp = 255;
            } else {
                exp = exp - 15 + 127;
            }

            union {
                uint32_t i;
                float f;
            } u = {sign | (exp << 23) | (frac << 13)};
            return u.f;
        }
    };

}  // namespace ndd
