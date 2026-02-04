#pragma once
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <mutex>

#if defined(USE_AVX512) || defined(USE_AVX2)
#    include <immintrin.h>
#endif

#if defined(USE_SVE2)
#    include <arm_sve.h>
#endif

#if defined(USE_NEON)
#    include <arm_neon.h>
#endif

namespace ndd {
    namespace quant {

        // Quantization level constants (compile-time constants for performance)
        enum class QuantizationLevel : uint8_t {
            FP32 = 32,   // Full precision float (4 bytes per dimension)
            INT16 = 16,  // Dynamic 16-bit integer quantization
            FP16 = 15,   // Half precision float (2 bytes per dimension)
            INT4 = 5,    // Dynamic 4-bit integer quantization
            BINARY = 1,  // Binary quantization (1 bit per dimension)
            INT8 = 8,    // Dynamic 8-bit integer quantization
            UNKNOWN = 0
        };

        // The "One Data Structure" that holds all behavior for a quantization level
        struct QuantizerDispatch {
            // Distance functions (void* allows generic usage by HNSW)
            float (*dist_l2)(const void* v1, const void* v2, const void* params);
            float (*dist_ip)(const void* v1, const void* v2, const void* params);
            float (*dist_cosine)(const void* v1, const void* v2, const void* params);

            // Similarity functions (for get_sim_func)
            float (*sim_l2)(const void* v1, const void* v2, const void* params);
            float (*sim_ip)(const void* v1, const void* v2, const void* params);
            float (*sim_cosine)(const void* v1, const void* v2, const void* params);

            // Conversion functions
            std::vector<uint8_t> (*quantize)(const std::vector<float>& in);
            std::vector<float> (*dequantize)(const uint8_t* in, size_t dim);

            // Direct quantization to INT8 (for mixed precision upper layers)
            std::vector<uint8_t> (*quantize_to_int8)(const void* in, size_t dim);

            // Metadata
            size_t (*get_storage_size)(size_t dim);
            float (*extract_inv_scale)(const uint8_t* in, size_t dim);
        };

        // Abstract base class for Quantization implementations
        class Quantizer {
        public:
            virtual ~Quantizer() = default;
            virtual std::string name() const = 0;
            virtual QuantizationLevel level() const = 0;
            virtual QuantizerDispatch getDispatch() const = 0;
        };

        // Singleton Registry for dynamic quantization support
        class QuantizationRegistry {
        public:
            static QuantizationRegistry& instance() {
                static QuantizationRegistry instance;
                return instance;
            }

            // Called by static initializers to register new types
            void registerQuantizer(QuantizationLevel level,
                                   const std::string& name,
                                   std::shared_ptr<Quantizer> impl = nullptr) {
                std::lock_guard<std::mutex> lock(mutex_);
                size_t idx = static_cast<size_t>(level);
                if(idx >= level_to_name_.size()) {
                    level_to_name_.resize(idx + 1);
                    quantizers_.resize(idx + 1);
                }
                level_to_name_[idx] = name;
                quantizers_[idx] = impl;
                name_to_level_[name] = level;
            }

            std::string toString(QuantizationLevel level) {
                size_t idx = static_cast<size_t>(level);
                std::lock_guard<std::mutex> lock(mutex_);
                if(idx < level_to_name_.size() && !level_to_name_[idx].empty()) {
                    return level_to_name_[idx];
                }
                return "unknown";
            }

            QuantizationLevel fromString(const std::string& str) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = name_to_level_.find(str);
                if(it != name_to_level_.end()) {
                    return it->second;
                }
                return QuantizationLevel::UNKNOWN;
            }

            // Return all registered names to allow dynamic validation
            std::vector<std::string> getRegisteredNames() {
                std::lock_guard<std::mutex> lock(mutex_);
                std::vector<std::string> names;
                names.reserve(name_to_level_.size());
                for(const auto& pair : name_to_level_) {
                    names.push_back(pair.first);
                }
                return names;
            }

            std::shared_ptr<Quantizer> getQuantizer(QuantizationLevel level) {
                size_t idx = static_cast<size_t>(level);
                std::lock_guard<std::mutex> lock(mutex_);
                if(idx < quantizers_.size()) {
                    return quantizers_[idx];
                }
                return nullptr;
            }

        private:
            QuantizationRegistry() {
                // No default registration to avoid circular dependencies.
                // Types will self-register.
            }

            std::mutex mutex_;
            std::map<std::string, QuantizationLevel> name_to_level_;
            std::vector<std::string> level_to_name_;
            std::vector<std::shared_ptr<Quantizer>> quantizers_;
        };

        // Helper struct for static registration in other files
        struct RegisterQuantizer {
            RegisterQuantizer(QuantizationLevel level,
                              const std::string& name,
                              std::shared_ptr<Quantizer> impl = nullptr) {
                QuantizationRegistry::instance().registerQuantizer(level, name, impl);
            }
        };

        inline std::string quantLevelToString(QuantizationLevel quant_level) {
            return QuantizationRegistry::instance().toString(quant_level);
        }

        inline QuantizationLevel stringToQuantLevel(const std::string& str) {
            return QuantizationRegistry::instance().fromString(str);
        }

        inline std::vector<std::string> getAvailableQuantizationNames() {
            return QuantizationRegistry::instance().getRegisteredNames();
        }

        // Get pointer to quantized data (before the scale)
        inline const void* get_quantized_data_ptr(const uint8_t* buffer) {
            return reinterpret_cast<const void*>(buffer);
        }

        namespace math {

            // Forward declarations for SIMD implementations
            inline float find_abs_max_scalar(const float* data, size_t size);
#if defined(USE_AVX512)
            inline float find_abs_max_avx512(const float* data, size_t size);
#endif
#if defined(USE_AVX2)
            inline float find_abs_max_avx2(const float* data, size_t size);
#endif
#if defined(USE_SVE2)
            inline float find_abs_max_sve(const float* data, size_t size);
#endif
#if defined(USE_NEON)
            inline float find_abs_max_neon(const float* data, size_t size);
#endif

            // Find absolute maximum value in a vector (for scaling)
            inline float find_abs_max(const float* data, size_t size) {
#if defined(USE_AVX512)
                return find_abs_max_avx512(data, size);
#elif defined(USE_SVE2)
                return find_abs_max_sve(data, size);
#elif defined(USE_AVX2)
                return find_abs_max_avx2(data, size);
#elif defined(USE_NEON)
                return find_abs_max_neon(data, size);
#else
                return find_abs_max_scalar(data, size);
#endif
            }

            // Scalar implementation for finding absolute maximum
            inline float find_abs_max_scalar(const float* data, size_t size) {
                float abs_max = 0.0f;
                for(size_t i = 0; i < size; ++i) {
                    abs_max = std::max(abs_max, std::abs(data[i]));
                }
                return abs_max;
            }

#if defined(USE_AVX512)
            // AVX512 optimized absolute maximum finding - MAXIMUM register utilization
            inline float find_abs_max_avx512(const float* data, size_t size) {
                if(size == 0) {
                    return 0.0f;
                }

                // Use 16 ZMM registers for parallel max finding (50% register utilization)
                // Keeping 16 registers free for compiler optimization and spills
                __m512 max_vec0 = _mm512_setzero_ps();
                __m512 max_vec1 = _mm512_setzero_ps();
                __m512 max_vec2 = _mm512_setzero_ps();
                __m512 max_vec3 = _mm512_setzero_ps();
                __m512 max_vec4 = _mm512_setzero_ps();
                __m512 max_vec5 = _mm512_setzero_ps();
                __m512 max_vec6 = _mm512_setzero_ps();
                __m512 max_vec7 = _mm512_setzero_ps();
                __m512 max_vec8 = _mm512_setzero_ps();
                __m512 max_vec9 = _mm512_setzero_ps();
                __m512 max_vec10 = _mm512_setzero_ps();
                __m512 max_vec11 = _mm512_setzero_ps();
                __m512 max_vec12 = _mm512_setzero_ps();
                __m512 max_vec13 = _mm512_setzero_ps();
                __m512 max_vec14 = _mm512_setzero_ps();
                __m512 max_vec15 = _mm512_setzero_ps();

                const __m512 sign_mask = _mm512_set1_ps(-0.0f);  // 0x80000000

                size_t i = 0;
                size_t vec_size =
                        (size / 256) * 256;  // Process 256 elements per iteration (16x unroll)

                // 16-way unrolled loop for maximum register utilization
                for(; i < vec_size; i += 256) {
                    // Load 16 vectors (256 floats total)
                    __m512 vec0 = _mm512_loadu_ps(&data[i]);
                    __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
                    __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
                    __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
                    __m512 vec4 = _mm512_loadu_ps(&data[i + 64]);
                    __m512 vec5 = _mm512_loadu_ps(&data[i + 80]);
                    __m512 vec6 = _mm512_loadu_ps(&data[i + 96]);
                    __m512 vec7 = _mm512_loadu_ps(&data[i + 112]);
                    __m512 vec8 = _mm512_loadu_ps(&data[i + 128]);
                    __m512 vec9 = _mm512_loadu_ps(&data[i + 144]);
                    __m512 vec10 = _mm512_loadu_ps(&data[i + 160]);
                    __m512 vec11 = _mm512_loadu_ps(&data[i + 176]);
                    __m512 vec12 = _mm512_loadu_ps(&data[i + 192]);
                    __m512 vec13 = _mm512_loadu_ps(&data[i + 208]);
                    __m512 vec14 = _mm512_loadu_ps(&data[i + 224]);
                    __m512 vec15 = _mm512_loadu_ps(&data[i + 240]);

                    // Clear sign bits (absolute value) for all 16 vectors
                    vec0 = _mm512_andnot_ps(sign_mask, vec0);
                    vec1 = _mm512_andnot_ps(sign_mask, vec1);
                    vec2 = _mm512_andnot_ps(sign_mask, vec2);
                    vec3 = _mm512_andnot_ps(sign_mask, vec3);
                    vec4 = _mm512_andnot_ps(sign_mask, vec4);
                    vec5 = _mm512_andnot_ps(sign_mask, vec5);
                    vec6 = _mm512_andnot_ps(sign_mask, vec6);
                    vec7 = _mm512_andnot_ps(sign_mask, vec7);
                    vec8 = _mm512_andnot_ps(sign_mask, vec8);
                    vec9 = _mm512_andnot_ps(sign_mask, vec9);
                    vec10 = _mm512_andnot_ps(sign_mask, vec10);
                    vec11 = _mm512_andnot_ps(sign_mask, vec11);
                    vec12 = _mm512_andnot_ps(sign_mask, vec12);
                    vec13 = _mm512_andnot_ps(sign_mask, vec13);
                    vec14 = _mm512_andnot_ps(sign_mask, vec14);
                    vec15 = _mm512_andnot_ps(sign_mask, vec15);

                    // Update max values for all 16 vectors in parallel
                    max_vec0 = _mm512_max_ps(max_vec0, vec0);
                    max_vec1 = _mm512_max_ps(max_vec1, vec1);
                    max_vec2 = _mm512_max_ps(max_vec2, vec2);
                    max_vec3 = _mm512_max_ps(max_vec3, vec3);
                    max_vec4 = _mm512_max_ps(max_vec4, vec4);
                    max_vec5 = _mm512_max_ps(max_vec5, vec5);
                    max_vec6 = _mm512_max_ps(max_vec6, vec6);
                    max_vec7 = _mm512_max_ps(max_vec7, vec7);
                    max_vec8 = _mm512_max_ps(max_vec8, vec8);
                    max_vec9 = _mm512_max_ps(max_vec9, vec9);
                    max_vec10 = _mm512_max_ps(max_vec10, vec10);
                    max_vec11 = _mm512_max_ps(max_vec11, vec11);
                    max_vec12 = _mm512_max_ps(max_vec12, vec12);
                    max_vec13 = _mm512_max_ps(max_vec13, vec13);
                    max_vec14 = _mm512_max_ps(max_vec14, vec14);
                    max_vec15 = _mm512_max_ps(max_vec15, vec15);
                }

                // Tree reduction of all max vectors
                max_vec0 = _mm512_max_ps(max_vec0, max_vec1);
                max_vec2 = _mm512_max_ps(max_vec2, max_vec3);
                max_vec4 = _mm512_max_ps(max_vec4, max_vec5);
                max_vec6 = _mm512_max_ps(max_vec6, max_vec7);
                max_vec8 = _mm512_max_ps(max_vec8, max_vec9);
                max_vec10 = _mm512_max_ps(max_vec10, max_vec11);
                max_vec12 = _mm512_max_ps(max_vec12, max_vec13);
                max_vec14 = _mm512_max_ps(max_vec14, max_vec15);

                max_vec0 = _mm512_max_ps(max_vec0, max_vec2);
                max_vec4 = _mm512_max_ps(max_vec4, max_vec6);
                max_vec8 = _mm512_max_ps(max_vec8, max_vec10);
                max_vec12 = _mm512_max_ps(max_vec12, max_vec14);

                max_vec0 = _mm512_max_ps(max_vec0, max_vec4);
                max_vec8 = _mm512_max_ps(max_vec8, max_vec12);

                __m512 final_max = _mm512_max_ps(max_vec0, max_vec8);

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (size / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m512 vec = _mm512_loadu_ps(&data[i]);
                    vec = _mm512_andnot_ps(sign_mask, vec);
                    final_max = _mm512_max_ps(final_max, vec);
                }

                // Horizontal reduction of final_max
                float result[16];
                _mm512_storeu_ps(result, final_max);
                float abs_max = 0.0f;
                for(int j = 0; j < 16; ++j) {
                    abs_max = std::max(abs_max, result[j]);
                }

                // Handle remaining elements
                for(; i < size; ++i) {
                    abs_max = std::max(abs_max, std::abs(data[i]));
                }

                return abs_max;
            }
#endif

#if defined(USE_AVX2)
            // AVX2 optimized absolute maximum finding
            inline float find_abs_max_avx2(const float* data, size_t size) {
                if(size == 0) {
                    return 0.0f;
                }

                // Use 16 YMM registers for parallel max finding
                __m256 max_vec0 = _mm256_setzero_ps();
                __m256 max_vec1 = _mm256_setzero_ps();
                __m256 max_vec2 = _mm256_setzero_ps();
                __m256 max_vec3 = _mm256_setzero_ps();
                __m256 max_vec4 = _mm256_setzero_ps();
                __m256 max_vec5 = _mm256_setzero_ps();
                __m256 max_vec6 = _mm256_setzero_ps();
                __m256 max_vec7 = _mm256_setzero_ps();
                __m256 max_vec8 = _mm256_setzero_ps();
                __m256 max_vec9 = _mm256_setzero_ps();
                __m256 max_vec10 = _mm256_setzero_ps();
                __m256 max_vec11 = _mm256_setzero_ps();
                __m256 max_vec12 = _mm256_setzero_ps();
                __m256 max_vec13 = _mm256_setzero_ps();
                __m256 max_vec14 = _mm256_setzero_ps();
                __m256 max_vec15 = _mm256_setzero_ps();

                const __m256 sign_mask = _mm256_set1_ps(-0.0f);  // 0x80000000

                size_t i = 0;
                size_t vec_size =
                        (size / 128)
                        * 128;  // Process 128 elements per iteration (16x unroll, 8 floats per YMM)

                for(; i < vec_size; i += 128) {
                    __m256 vec0 = _mm256_loadu_ps(&data[i]);
                    __m256 vec1 = _mm256_loadu_ps(&data[i + 8]);
                    __m256 vec2 = _mm256_loadu_ps(&data[i + 16]);
                    __m256 vec3 = _mm256_loadu_ps(&data[i + 24]);
                    __m256 vec4 = _mm256_loadu_ps(&data[i + 32]);
                    __m256 vec5 = _mm256_loadu_ps(&data[i + 40]);
                    __m256 vec6 = _mm256_loadu_ps(&data[i + 48]);
                    __m256 vec7 = _mm256_loadu_ps(&data[i + 56]);
                    __m256 vec8 = _mm256_loadu_ps(&data[i + 64]);
                    __m256 vec9 = _mm256_loadu_ps(&data[i + 72]);
                    __m256 vec10 = _mm256_loadu_ps(&data[i + 80]);
                    __m256 vec11 = _mm256_loadu_ps(&data[i + 88]);
                    __m256 vec12 = _mm256_loadu_ps(&data[i + 96]);
                    __m256 vec13 = _mm256_loadu_ps(&data[i + 104]);
                    __m256 vec14 = _mm256_loadu_ps(&data[i + 112]);
                    __m256 vec15 = _mm256_loadu_ps(&data[i + 120]);

                    vec0 = _mm256_andnot_ps(sign_mask, vec0);
                    vec1 = _mm256_andnot_ps(sign_mask, vec1);
                    vec2 = _mm256_andnot_ps(sign_mask, vec2);
                    vec3 = _mm256_andnot_ps(sign_mask, vec3);
                    vec4 = _mm256_andnot_ps(sign_mask, vec4);
                    vec5 = _mm256_andnot_ps(sign_mask, vec5);
                    vec6 = _mm256_andnot_ps(sign_mask, vec6);
                    vec7 = _mm256_andnot_ps(sign_mask, vec7);
                    vec8 = _mm256_andnot_ps(sign_mask, vec8);
                    vec9 = _mm256_andnot_ps(sign_mask, vec9);
                    vec10 = _mm256_andnot_ps(sign_mask, vec10);
                    vec11 = _mm256_andnot_ps(sign_mask, vec11);
                    vec12 = _mm256_andnot_ps(sign_mask, vec12);
                    vec13 = _mm256_andnot_ps(sign_mask, vec13);
                    vec14 = _mm256_andnot_ps(sign_mask, vec14);
                    vec15 = _mm256_andnot_ps(sign_mask, vec15);

                    max_vec0 = _mm256_max_ps(max_vec0, vec0);
                    max_vec1 = _mm256_max_ps(max_vec1, vec1);
                    max_vec2 = _mm256_max_ps(max_vec2, vec2);
                    max_vec3 = _mm256_max_ps(max_vec3, vec3);
                    max_vec4 = _mm256_max_ps(max_vec4, vec4);
                    max_vec5 = _mm256_max_ps(max_vec5, vec5);
                    max_vec6 = _mm256_max_ps(max_vec6, vec6);
                    max_vec7 = _mm256_max_ps(max_vec7, vec7);
                    max_vec8 = _mm256_max_ps(max_vec8, vec8);
                    max_vec9 = _mm256_max_ps(max_vec9, vec9);
                    max_vec10 = _mm256_max_ps(max_vec10, vec10);
                    max_vec11 = _mm256_max_ps(max_vec11, vec11);
                    max_vec12 = _mm256_max_ps(max_vec12, vec12);
                    max_vec13 = _mm256_max_ps(max_vec13, vec13);
                    max_vec14 = _mm256_max_ps(max_vec14, vec14);
                    max_vec15 = _mm256_max_ps(max_vec15, vec15);
                }

                // Tree reduction
                max_vec0 = _mm256_max_ps(max_vec0, max_vec1);
                max_vec2 = _mm256_max_ps(max_vec2, max_vec3);
                max_vec4 = _mm256_max_ps(max_vec4, max_vec5);
                max_vec6 = _mm256_max_ps(max_vec6, max_vec7);
                max_vec8 = _mm256_max_ps(max_vec8, max_vec9);
                max_vec10 = _mm256_max_ps(max_vec10, max_vec11);
                max_vec12 = _mm256_max_ps(max_vec12, max_vec13);
                max_vec14 = _mm256_max_ps(max_vec14, max_vec15);

                max_vec0 = _mm256_max_ps(max_vec0, max_vec2);
                max_vec4 = _mm256_max_ps(max_vec4, max_vec6);
                max_vec8 = _mm256_max_ps(max_vec8, max_vec10);
                max_vec12 = _mm256_max_ps(max_vec12, max_vec14);

                max_vec0 = _mm256_max_ps(max_vec0, max_vec4);
                max_vec8 = _mm256_max_ps(max_vec8, max_vec12);

                __m256 final_max = _mm256_max_ps(max_vec0, max_vec8);

                // Handle remaining 8-element chunks
                size_t remaining_vec_size = (size / 8) * 8;
                for(; i < remaining_vec_size; i += 8) {
                    __m256 vec = _mm256_loadu_ps(&data[i]);
                    vec = _mm256_andnot_ps(sign_mask, vec);
                    final_max = _mm256_max_ps(final_max, vec);
                }

                // Horizontal reduction
                float result[8];
                _mm256_storeu_ps(result, final_max);
                float abs_max = 0.0f;
                for(int j = 0; j < 8; ++j) {
                    abs_max = std::max(abs_max, result[j]);
                }

                // Handle remaining elements
                for(; i < size; ++i) {
                    abs_max = std::max(abs_max, std::abs(data[i]));
                }

                return abs_max;
            }
#endif

#if defined(USE_SVE2)
            // SVE2 optimized absolute maximum finding
            inline float find_abs_max_sve(const float* data, size_t size) {
                if(size == 0) {
                    return 0.0f;
                }

                svbool_t pg = svptrue_b32();
                svfloat32_t max_val = svdup_f32(0.0f);

                size_t i = 0;
                size_t num_elements = svcntw();

                // Unroll 4 times for SVE
                size_t vec_size = (size / (num_elements * 4)) * (num_elements * 4);

                for(; i < vec_size; i += num_elements * 4) {
                    svfloat32_t vec0 = svld1_f32(pg, &data[i]);
                    svfloat32_t vec1 = svld1_f32(pg, &data[i + num_elements]);
                    svfloat32_t vec2 = svld1_f32(pg, &data[i + num_elements * 2]);
                    svfloat32_t vec3 = svld1_f32(pg, &data[i + num_elements * 3]);

                    vec0 = svabs_f32_x(pg, vec0);
                    vec1 = svabs_f32_x(pg, vec1);
                    vec2 = svabs_f32_x(pg, vec2);
                    vec3 = svabs_f32_x(pg, vec3);

                    max_val = svmax_f32_x(pg, max_val, vec0);
                    max_val = svmax_f32_x(pg, max_val, vec1);
                    max_val = svmax_f32_x(pg, max_val, vec2);
                    max_val = svmax_f32_x(pg, max_val, vec3);
                }

                // Handle remaining vectors
                while(i + num_elements <= size) {
                    svfloat32_t vec = svld1_f32(pg, &data[i]);
                    vec = svabs_f32_x(pg, vec);
                    max_val = svmax_f32_x(pg, max_val, vec);
                    i += num_elements;
                }

                float abs_max = svmaxv_f32(pg, max_val);

                // Handle remaining elements with predicate
                if(i < size) {
                    pg = svwhilelt_b32(i, size);
                    svfloat32_t vec = svld1_f32(pg, &data[i]);
                    vec = svabs_f32_x(pg, vec);
                    float partial_max = svmaxv_f32(pg, vec);
                    abs_max = std::max(abs_max, partial_max);
                }

                return abs_max;
            }
#endif

#if defined(USE_NEON)
            // NEON optimized absolute maximum finding - MAXIMUM register utilization
            inline float find_abs_max_neon(const float* data, size_t size) {
                if(size == 0) {
                    return 0.0f;
                }

                // Use 16 NEON registers for parallel max finding (50% register utilization)
                // Keeping 16 registers free for compiler optimization and spills
                float32x4_t max_vec0 = vdupq_n_f32(0.0f);
                float32x4_t max_vec1 = vdupq_n_f32(0.0f);
                float32x4_t max_vec2 = vdupq_n_f32(0.0f);
                float32x4_t max_vec3 = vdupq_n_f32(0.0f);
                float32x4_t max_vec4 = vdupq_n_f32(0.0f);
                float32x4_t max_vec5 = vdupq_n_f32(0.0f);
                float32x4_t max_vec6 = vdupq_n_f32(0.0f);
                float32x4_t max_vec7 = vdupq_n_f32(0.0f);
                float32x4_t max_vec8 = vdupq_n_f32(0.0f);
                float32x4_t max_vec9 = vdupq_n_f32(0.0f);
                float32x4_t max_vec10 = vdupq_n_f32(0.0f);
                float32x4_t max_vec11 = vdupq_n_f32(0.0f);
                float32x4_t max_vec12 = vdupq_n_f32(0.0f);
                float32x4_t max_vec13 = vdupq_n_f32(0.0f);
                float32x4_t max_vec14 = vdupq_n_f32(0.0f);
                float32x4_t max_vec15 = vdupq_n_f32(0.0f);

                size_t i = 0;
                size_t vec_size =
                        (size / 64) * 64;  // Process 64 elements per iteration (16x unroll)

                // 16-way unrolled loop for maximum register utilization
                for(; i < vec_size; i += 64) {
                    // Load 16 vectors (64 floats total)
                    float32x4_t vec0 = vld1q_f32(&data[i]);
                    float32x4_t vec1 = vld1q_f32(&data[i + 4]);
                    float32x4_t vec2 = vld1q_f32(&data[i + 8]);
                    float32x4_t vec3 = vld1q_f32(&data[i + 12]);
                    float32x4_t vec4 = vld1q_f32(&data[i + 16]);
                    float32x4_t vec5 = vld1q_f32(&data[i + 20]);
                    float32x4_t vec6 = vld1q_f32(&data[i + 24]);
                    float32x4_t vec7 = vld1q_f32(&data[i + 28]);
                    float32x4_t vec8 = vld1q_f32(&data[i + 32]);
                    float32x4_t vec9 = vld1q_f32(&data[i + 36]);
                    float32x4_t vec10 = vld1q_f32(&data[i + 40]);
                    float32x4_t vec11 = vld1q_f32(&data[i + 44]);
                    float32x4_t vec12 = vld1q_f32(&data[i + 48]);
                    float32x4_t vec13 = vld1q_f32(&data[i + 52]);
                    float32x4_t vec14 = vld1q_f32(&data[i + 56]);
                    float32x4_t vec15 = vld1q_f32(&data[i + 60]);

                    // Absolute value for all 16 vectors
                    vec0 = vabsq_f32(vec0);
                    vec1 = vabsq_f32(vec1);
                    vec2 = vabsq_f32(vec2);
                    vec3 = vabsq_f32(vec3);
                    vec4 = vabsq_f32(vec4);
                    vec5 = vabsq_f32(vec5);
                    vec6 = vabsq_f32(vec6);
                    vec7 = vabsq_f32(vec7);
                    vec8 = vabsq_f32(vec8);
                    vec9 = vabsq_f32(vec9);
                    vec10 = vabsq_f32(vec10);
                    vec11 = vabsq_f32(vec11);
                    vec12 = vabsq_f32(vec12);
                    vec13 = vabsq_f32(vec13);
                    vec14 = vabsq_f32(vec14);
                    vec15 = vabsq_f32(vec15);

                    // Update max values for all 16 vectors in parallel
                    max_vec0 = vmaxq_f32(max_vec0, vec0);
                    max_vec1 = vmaxq_f32(max_vec1, vec1);
                    max_vec2 = vmaxq_f32(max_vec2, vec2);
                    max_vec3 = vmaxq_f32(max_vec3, vec3);
                    max_vec4 = vmaxq_f32(max_vec4, vec4);
                    max_vec5 = vmaxq_f32(max_vec5, vec5);
                    max_vec6 = vmaxq_f32(max_vec6, vec6);
                    max_vec7 = vmaxq_f32(max_vec7, vec7);
                    max_vec8 = vmaxq_f32(max_vec8, vec8);
                    max_vec9 = vmaxq_f32(max_vec9, vec9);
                    max_vec10 = vmaxq_f32(max_vec10, vec10);
                    max_vec11 = vmaxq_f32(max_vec11, vec11);
                    max_vec12 = vmaxq_f32(max_vec12, vec12);
                    max_vec13 = vmaxq_f32(max_vec13, vec13);
                    max_vec14 = vmaxq_f32(max_vec14, vec14);
                    max_vec15 = vmaxq_f32(max_vec15, vec15);
                }

                // Tree reduction of all max vectors
                max_vec0 = vmaxq_f32(max_vec0, max_vec1);
                max_vec2 = vmaxq_f32(max_vec2, max_vec3);
                max_vec4 = vmaxq_f32(max_vec4, max_vec5);
                max_vec6 = vmaxq_f32(max_vec6, max_vec7);
                max_vec8 = vmaxq_f32(max_vec8, max_vec9);
                max_vec10 = vmaxq_f32(max_vec10, max_vec11);
                max_vec12 = vmaxq_f32(max_vec12, max_vec13);
                max_vec14 = vmaxq_f32(max_vec14, max_vec15);

                max_vec0 = vmaxq_f32(max_vec0, max_vec2);
                max_vec4 = vmaxq_f32(max_vec4, max_vec6);
                max_vec8 = vmaxq_f32(max_vec8, max_vec10);
                max_vec12 = vmaxq_f32(max_vec12, max_vec14);

                max_vec0 = vmaxq_f32(max_vec0, max_vec4);
                max_vec8 = vmaxq_f32(max_vec8, max_vec12);

                float32x4_t final_max = vmaxq_f32(max_vec0, max_vec8);

                // Handle remaining 4-element chunks
                size_t remaining_vec_size = (size / 4) * 4;
                for(; i < remaining_vec_size; i += 4) {
                    float32x4_t vec = vld1q_f32(&data[i]);
                    vec = vabsq_f32(vec);
                    final_max = vmaxq_f32(final_max, vec);
                }

                // Horizontal reduction of final_max
                float32x2_t max_pair = vmax_f32(vget_low_f32(final_max), vget_high_f32(final_max));
                max_pair = vpmax_f32(max_pair, max_pair);
                float abs_max = vget_lane_f32(max_pair, 0);

                // Handle remaining elements
                for(; i < size; ++i) {
                    abs_max = std::max(abs_max, std::abs(data[i]));
                }

                return abs_max;
            }
#endif

        }  // namespace math

    }  // namespace quant
}  // namespace ndd
