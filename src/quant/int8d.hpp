#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "common.hpp"
#include "../hnsw/hnswlib.h"

namespace ndd {
    namespace quant {
        namespace int8d {

            constexpr float INT8_SCALE = 127.0f;  // Max value for 8-bit signed integer quantization

            constexpr size_t get_storage_size(size_t dimension) {
                return dimension * sizeof(int8_t) + sizeof(float);
            }

            inline float extract_inv_scale(const uint8_t* buffer, size_t dimension) {
                return *reinterpret_cast<const float*>(buffer + dimension * sizeof(int8_t));
            }

            // Quantize FP32 vector to INT8 + scale, store in uint8_t buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = ndd::quant::math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;  // Avoid division by zero
                }
                float scale = INT8_SCALE / abs_max;

                // Quantize data
                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                for(size_t i = 0; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int8_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP8_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = abs_max / INT8_SCALE;

                return buffer;
            }

#if defined(USE_AVX512)
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_avx512(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = ndd::quant::math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT8_SCALE / abs_max;

                // SIMD quantization - using more registers for better parallelism
                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                const __m512 scale_vec = _mm512_set1_ps(scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 64) * 64;  // Process 64 floats -> 64 int8s (4x unrolling)

                // 4-way unrolled loop using 12 ZMM registers for better ILP
                for(; i < vec_size; i += 64) {
                    // Load 4 vectors (64 floats total)
                    __m512 vec0 = _mm512_loadu_ps(&input[i]);
                    __m512 vec1 = _mm512_loadu_ps(&input[i + 16]);
                    __m512 vec2 = _mm512_loadu_ps(&input[i + 32]);
                    __m512 vec3 = _mm512_loadu_ps(&input[i + 48]);

                    // Scale all 4 vectors in parallel
                    vec0 = _mm512_mul_ps(vec0, scale_vec);
                    vec1 = _mm512_mul_ps(vec1, scale_vec);
                    vec2 = _mm512_mul_ps(vec2, scale_vec);
                    vec3 = _mm512_mul_ps(vec3, scale_vec);

                    // Convert to int32 (with rounding)
                    __m512i int_vec0 = _mm512_cvtps_epi32(vec0);
                    __m512i int_vec1 = _mm512_cvtps_epi32(vec1);
                    __m512i int_vec2 = _mm512_cvtps_epi32(vec2);
                    __m512i int_vec3 = _mm512_cvtps_epi32(vec3);

                    // Pack to int8 (64 int32 -> 64 int8) - process in 4 groups of 16
                    __m128i packed0 = _mm512_cvtepi32_epi8(int_vec0);
                    __m128i packed1 = _mm512_cvtepi32_epi8(int_vec1);
                    __m128i packed2 = _mm512_cvtepi32_epi8(int_vec2);
                    __m128i packed3 = _mm512_cvtepi32_epi8(int_vec3);

                    // Store all 4 packed vectors (each is 16 int8 values)
                    _mm_storeu_si128((__m128i*)&data_ptr[i], packed0);
                    _mm_storeu_si128((__m128i*)&data_ptr[i + 16], packed1);
                    _mm_storeu_si128((__m128i*)&data_ptr[i + 32], packed2);
                    _mm_storeu_si128((__m128i*)&data_ptr[i + 48], packed3);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (dimension / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m512 vec = _mm512_loadu_ps(&input[i]);
                    vec = _mm512_mul_ps(vec, scale_vec);

                    __m512i int_vec = _mm512_cvtps_epi32(vec);
                    __m128i packed = _mm512_cvtepi32_epi8(int_vec);
                    _mm_storeu_si128((__m128i*)&data_ptr[i], packed);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int8_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP8_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = abs_max / INT8_SCALE;

                return buffer;
            }
#endif

#if defined(USE_NEON)
            // NEON optimized quantization FP32 -> INT8 buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_neon(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = ndd::quant::math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT8_SCALE / abs_max;

                // SIMD quantization - using more registers for better parallelism
                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                const float32x4_t scale_vec = vdupq_n_f32(scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 32) * 32;  // Process 32 floats -> 32 int8s (2x unrolling)

                // 2-way unrolled loop using 16 NEON registers for better ILP
                for(; i < vec_size; i += 32) {
                    // Load 8 vectors (32 floats total)
                    float32x4_t vec0 = vld1q_f32(&input[i]);
                    float32x4_t vec1 = vld1q_f32(&input[i + 4]);
                    float32x4_t vec2 = vld1q_f32(&input[i + 8]);
                    float32x4_t vec3 = vld1q_f32(&input[i + 12]);
                    float32x4_t vec4 = vld1q_f32(&input[i + 16]);
                    float32x4_t vec5 = vld1q_f32(&input[i + 20]);
                    float32x4_t vec6 = vld1q_f32(&input[i + 24]);
                    float32x4_t vec7 = vld1q_f32(&input[i + 28]);

                    // Scale all 8 vectors in parallel
                    vec0 = vmulq_f32(vec0, scale_vec);
                    vec1 = vmulq_f32(vec1, scale_vec);
                    vec2 = vmulq_f32(vec2, scale_vec);
                    vec3 = vmulq_f32(vec3, scale_vec);
                    vec4 = vmulq_f32(vec4, scale_vec);
                    vec5 = vmulq_f32(vec5, scale_vec);
                    vec6 = vmulq_f32(vec6, scale_vec);
                    vec7 = vmulq_f32(vec7, scale_vec);

                    // Convert to int32 (with rounding)
                    int32x4_t int_vec0 = vcvtaq_s32_f32(vec0);
                    int32x4_t int_vec1 = vcvtaq_s32_f32(vec1);
                    int32x4_t int_vec2 = vcvtaq_s32_f32(vec2);
                    int32x4_t int_vec3 = vcvtaq_s32_f32(vec3);
                    int32x4_t int_vec4 = vcvtaq_s32_f32(vec4);
                    int32x4_t int_vec5 = vcvtaq_s32_f32(vec5);
                    int32x4_t int_vec6 = vcvtaq_s32_f32(vec6);
                    int32x4_t int_vec7 = vcvtaq_s32_f32(vec7);

                    // Pack to int16 (32 int32 -> 32 int16)
                    int16x8_t packed01 = vcombine_s16(vqmovn_s32(int_vec0), vqmovn_s32(int_vec1));
                    int16x8_t packed23 = vcombine_s16(vqmovn_s32(int_vec2), vqmovn_s32(int_vec3));
                    int16x8_t packed45 = vcombine_s16(vqmovn_s32(int_vec4), vqmovn_s32(int_vec5));
                    int16x8_t packed67 = vcombine_s16(vqmovn_s32(int_vec6), vqmovn_s32(int_vec7));

                    // Pack to int8 (32 int16 -> 32 int8)
                    int8x16_t final_packed0 =
                            vcombine_s8(vqmovn_s16(packed01), vqmovn_s16(packed23));
                    int8x16_t final_packed1 =
                            vcombine_s8(vqmovn_s16(packed45), vqmovn_s16(packed67));

                    // Store both packed vectors
                    vst1q_s8(&data_ptr[i], final_packed0);
                    vst1q_s8(&data_ptr[i + 16], final_packed1);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (dimension / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    float32x4_t vec1 = vld1q_f32(&input[i]);
                    float32x4_t vec2 = vld1q_f32(&input[i + 4]);
                    float32x4_t vec3 = vld1q_f32(&input[i + 8]);
                    float32x4_t vec4 = vld1q_f32(&input[i + 12]);

                    vec1 = vmulq_f32(vec1, scale_vec);
                    vec2 = vmulq_f32(vec2, scale_vec);
                    vec3 = vmulq_f32(vec3, scale_vec);
                    vec4 = vmulq_f32(vec4, scale_vec);

                    int32x4_t int_vec1 = vcvtaq_s32_f32(vec1);
                    int32x4_t int_vec2 = vcvtaq_s32_f32(vec2);
                    int32x4_t int_vec3 = vcvtaq_s32_f32(vec3);
                    int32x4_t int_vec4 = vcvtaq_s32_f32(vec4);

                    int16x8_t packed12 = vcombine_s16(vqmovn_s32(int_vec1), vqmovn_s32(int_vec2));
                    int16x8_t packed34 = vcombine_s16(vqmovn_s32(int_vec3), vqmovn_s32(int_vec4));

                    int8x16_t final_packed =
                            vcombine_s8(vqmovn_s16(packed12), vqmovn_s16(packed34));
                    vst1q_s8(&data_ptr[i], final_packed);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int8_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP8_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = abs_max / INT8_SCALE;

                return buffer;
            }
#endif

#if defined(USE_SVE2)
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_sve(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                float abs_max = ndd::quant::math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT8_SCALE / abs_max;

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());

                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dimension);

                while(svptest_any(svptrue_b32(), pg)) {
                    svfloat32_t vec = svld1_f32(pg, &input[i]);
                    vec = svmul_f32_x(pg, vec, svdup_f32(scale));

                    // Round to nearest integer, ties away from zero (matches std::round/vcvta)
                    vec = svrinta_f32_x(pg, vec);

                    svint32_t int_vec = svcvt_s32_f32_x(pg, vec);

                    // Saturate to symmetric int8 range [-127, 127]
                    // We explicitly avoid -128 to ensure symmetry and match the scaling factor
                    // 127.0/abs_max
                    int_vec = svmin_s32_x(pg, int_vec, svdup_s32(127));
                    int_vec = svmax_s32_x(pg, int_vec, svdup_s32(-127));

                    // Store lower 8 bits
                    svst1b_s32(pg, &data_ptr[i], int_vec);

                    i += svcntw();
                    pg = svwhilelt_b32(i, dimension);
                }

                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = abs_max / INT8_SCALE;

                return buffer;
            }
#endif

#if defined(USE_AVX2)
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_avx2(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Max
                float abs_max = 0.0f;
                size_t i = 0;
                __m256 max_vec = _mm256_setzero_ps();
                static const __m256 sign_mask_vec = _mm256_set1_ps(-0.0f);

                for(; i + 8 <= dimension; i += 8) {
                    __m256 v = _mm256_loadu_ps(&input[i]);
                    v = _mm256_andnot_ps(sign_mask_vec, v);
                    max_vec = _mm256_max_ps(max_vec, v);
                }

                __m128 max128 = _mm_max_ps(_mm256_castsi256_si128(_mm256_castps_si256(max_vec)),
                                           _mm256_extractf128_ps(max_vec, 1));
                max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
                max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0x1));
                abs_max = _mm_cvtss_f32(max128);

                for(; i < dimension; ++i) {
                    abs_max = std::max(abs_max, std::abs(input[i]));
                }

                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT8_SCALE / abs_max;

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                __m256 scale_vec = _mm256_set1_ps(scale);

                i = 0;
                for(; i + 32 <= dimension; i += 32) {
                    __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(&input[i]), scale_vec);
                    __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(&input[i + 8]), scale_vec);
                    __m256 v2 = _mm256_mul_ps(_mm256_loadu_ps(&input[i + 16]), scale_vec);
                    __m256 v3 = _mm256_mul_ps(_mm256_loadu_ps(&input[i + 24]), scale_vec);

                    __m256i i0 = _mm256_cvtps_epi32(v0);
                    __m256i i1 = _mm256_cvtps_epi32(v1);
                    __m256i i2 = _mm256_cvtps_epi32(v2);
                    __m256i i3 = _mm256_cvtps_epi32(v3);

                    // Pack to 16-bit (interleaved lanes)
                    __m256i p01 = _mm256_packs_epi32(i0, i1);
                    __m256i p23 = _mm256_packs_epi32(i2, i3);

                    // De-interleave 16-bit
                    p01 = _mm256_permute4x64_epi64(p01, _MM_SHUFFLE(3, 1, 2, 0));
                    p23 = _mm256_permute4x64_epi64(p23, _MM_SHUFFLE(3, 1, 2, 0));

                    // Pack to 8-bit (produces interleaved lanes [v0, v2 | v1, v3] 64-bit blocks)
                    __m256i p = _mm256_packs_epi16(p01, p23);

                    // Fix order
                    p = _mm256_permute4x64_epi64(p, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm256_storeu_si256((__m256i*)(data_ptr + i), p);
                }

                for(; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int8_t>(std::round(scaled));
                }

                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = abs_max / INT8_SCALE;

                return buffer;
            }
#endif

            // Auto-select best quantization implementation for INT8 -> uint8_t buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_auto(const std::vector<float>& input) {
#if defined(USE_AVX512)
                return quantize_vector_fp32_to_int8_buffer_avx512(input);
#elif defined(USE_SVE2)
                return quantize_vector_fp32_to_int8_buffer_sve(input);
#elif defined(USE_AVX2)
                return quantize_vector_fp32_to_int8_buffer_avx2(input);
#elif defined(USE_NEON)
                return quantize_vector_fp32_to_int8_buffer_neon(input);
#else
                return quantize_vector_fp32_to_int8_buffer(input);
#endif
            }

#if defined(USE_AVX512)
            // AVX512 optimized dequantization INT8 buffer -> FP32 vector
            inline std::vector<float> dequantize_int8_buffer_to_fp32_avx512(const uint8_t* buffer,
                                                                            size_t dimension) {
                std::vector<float> output(dimension);
                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 64) * 64;  // Process 64 int8s -> 64 floats (4x unrolling)

                // 4-way unrolled loop using multiple ZMM registers
                for(; i < vec_size; i += 64) {
                    // Load 64 int8 values (4 x 128-bit loads)
                    __m128i int8_vec0 = _mm_loadu_si128((__m128i*)&data_ptr[i]);
                    __m128i int8_vec1 = _mm_loadu_si128((__m128i*)&data_ptr[i + 16]);
                    __m128i int8_vec2 = _mm_loadu_si128((__m128i*)&data_ptr[i + 32]);
                    __m128i int8_vec3 = _mm_loadu_si128((__m128i*)&data_ptr[i + 48]);

                    // Convert to int32 (sign extend)
                    __m512i int32_vec0 = _mm512_cvtepi8_epi32(int8_vec0);
                    __m512i int32_vec1 = _mm512_cvtepi8_epi32(int8_vec1);
                    __m512i int32_vec2 = _mm512_cvtepi8_epi32(int8_vec2);
                    __m512i int32_vec3 = _mm512_cvtepi8_epi32(int8_vec3);

                    // Convert to float
                    __m512 float_vec0 = _mm512_cvtepi32_ps(int32_vec0);
                    __m512 float_vec1 = _mm512_cvtepi32_ps(int32_vec1);
                    __m512 float_vec2 = _mm512_cvtepi32_ps(int32_vec2);
                    __m512 float_vec3 = _mm512_cvtepi32_ps(int32_vec3);

                    // Apply inverse scale
                    float_vec0 = _mm512_mul_ps(float_vec0, inv_scale_vec);
                    float_vec1 = _mm512_mul_ps(float_vec1, inv_scale_vec);
                    float_vec2 = _mm512_mul_ps(float_vec2, inv_scale_vec);
                    float_vec3 = _mm512_mul_ps(float_vec3, inv_scale_vec);

                    // Store results
                    _mm512_storeu_ps(&output[i], float_vec0);
                    _mm512_storeu_ps(&output[i + 16], float_vec1);
                    _mm512_storeu_ps(&output[i + 32], float_vec2);
                    _mm512_storeu_ps(&output[i + 48], float_vec3);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (dimension / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m128i int8_vec = _mm_loadu_si128((__m128i*)&data_ptr[i]);
                    __m512i int32_vec = _mm512_cvtepi8_epi32(int8_vec);
                    __m512 float_vec = _mm512_cvtepi32_ps(int32_vec);
                    float_vec = _mm512_mul_ps(float_vec, inv_scale_vec);
                    _mm512_storeu_ps(&output[i], float_vec);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    output[i] = static_cast<float>(data_ptr[i]) / scale;
                }

                return output;
            }
#endif

#if defined(USE_NEON)
            inline std::vector<float> dequantize_int8_buffer_to_fp32_neon(const uint8_t* buffer,
                                                                          size_t dimension) {
                std::vector<float> output(dimension);
                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                const float32x4_t inv_scale_vec = vdupq_n_f32(1.0f / scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 16) * 16;  // Process 16 int8s -> 16 floats (4x unrolling)

                for(; i < vec_size; i += 16) {
                    // Load 16 int8 values
                    int8x16_t int8_vec = vld1q_s8(&data_ptr[i]);

                    // Convert to int16 (low and high parts)
                    int16x8_t int16_low = vmovl_s8(vget_low_s8(int8_vec));
                    int16x8_t int16_high = vmovl_s8(vget_high_s8(int8_vec));

                    // Convert to int32 (4 parts)
                    int32x4_t int32_0 = vmovl_s16(vget_low_s16(int16_low));
                    int32x4_t int32_1 = vmovl_s16(vget_high_s16(int16_low));
                    int32x4_t int32_2 = vmovl_s16(vget_low_s16(int16_high));
                    int32x4_t int32_3 = vmovl_s16(vget_high_s16(int16_high));

                    // Convert to float
                    float32x4_t float_0 = vcvtq_f32_s32(int32_0);
                    float32x4_t float_1 = vcvtq_f32_s32(int32_1);
                    float32x4_t float_2 = vcvtq_f32_s32(int32_2);
                    float32x4_t float_3 = vcvtq_f32_s32(int32_3);

                    // Apply inverse scale
                    float_0 = vmulq_f32(float_0, inv_scale_vec);
                    float_1 = vmulq_f32(float_1, inv_scale_vec);
                    float_2 = vmulq_f32(float_2, inv_scale_vec);
                    float_3 = vmulq_f32(float_3, inv_scale_vec);

                    // Store results
                    vst1q_f32(&output[i], float_0);
                    vst1q_f32(&output[i + 4], float_1);
                    vst1q_f32(&output[i + 8], float_2);
                    vst1q_f32(&output[i + 12], float_3);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    output[i] = static_cast<float>(data_ptr[i]) / scale;
                }

                return output;
            }
#endif

#if defined(USE_SVE2)
            inline std::vector<float> dequantize_int8_buffer_to_fp32_sve(const uint8_t* buffer,
                                                                         size_t dimension) {
                std::vector<float> output(dimension);
                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dimension);

                while(svptest_any(svptrue_b32(), pg)) {
                    svint32_t int_vec = svld1sb_s32(pg, &data_ptr[i]);
                    svfloat32_t float_vec = svcvt_f32_s32_x(pg, int_vec);
                    float_vec = svmul_f32_x(pg, float_vec, svdup_f32(1.0f / scale));
                    svst1_f32(pg, &output[i], float_vec);

                    i += svcntw();
                    pg = svwhilelt_b32(i, dimension);
                }
                return output;
            }
#endif

            // Auto-select best dequantization implementation for INT8 buffer -> FP32 vector
            inline std::vector<float> dequantize_int8_buffer_to_fp32(const uint8_t* buffer,
                                                                     size_t dimension) {
#if defined(USE_AVX512)
                return dequantize_int8_buffer_to_fp32_avx512(buffer, dimension);
#elif defined(USE_SVE2)
                return dequantize_int8_buffer_to_fp32_sve(buffer, dimension);
#elif defined(USE_NEON)
                return dequantize_int8_buffer_to_fp32_neon(buffer, dimension);
#else
                // Scalar fallback
                std::vector<float> output(dimension);
                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                for(size_t i = 0; i < dimension; ++i) {
                    output[i] = static_cast<float>(data_ptr[i]) / scale;
                }
                return output;
#endif
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int8_buffer_auto(input);
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                return dequantize_int8_buffer_to_fp32(in, dim);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = (const int8_t*)pVect1v;
                const int8_t* pVect2 = (const int8_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                size_t qty = params->dim;

                float scale1 = extract_inv_scale((const uint8_t*)pVect1, qty);
                float scale2 = extract_inv_scale((const uint8_t*)pVect2, qty);

                float res = 0;
                size_t i = 0;

#if defined(USE_AVX512)
                __m512 sum = _mm512_setzero_ps();
                __m512 v_scale1 = _mm512_set1_ps(scale1);
                __m512 v_scale2 = _mm512_set1_ps(scale2);

                for(; i + 16 <= qty; i += 16) {
                    __m128i v1_i8 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
                    __m128i v2_i8 = _mm_loadu_si128((const __m128i*)(pVect2 + i));

                    __m512i v1_i32 = _mm512_cvtepi8_epi32(v1_i8);
                    __m512i v2_i32 = _mm512_cvtepi8_epi32(v2_i8);

                    __m512 v1_f = _mm512_cvtepi32_ps(v1_i32);
                    __m512 v2_f = _mm512_cvtepi32_ps(v2_i32);

                    v1_f = _mm512_mul_ps(v1_f, v_scale1);
                    v2_f = _mm512_mul_ps(v2_f, v_scale2);

                    __m512 diff = _mm512_sub_ps(v1_f, v2_f);
                    sum = _mm512_fmadd_ps(diff, diff, sum);
                }
                res = _mm512_reduce_add_ps(sum);
#elif defined(USE_AVX2)
                __m256 sum = _mm256_setzero_ps();
                __m256 v_scale1 = _mm256_set1_ps(scale1);
                __m256 v_scale2 = _mm256_set1_ps(scale2);

                for(; i + 8 <= qty; i += 8) {
                    __m128i v1_i8 = _mm_loadl_epi64((const __m128i*)(pVect1 + i));
                    __m128i v2_i8 = _mm_loadl_epi64((const __m128i*)(pVect2 + i));

                    __m256i v1_i32 = _mm256_cvtepi8_epi32(v1_i8);
                    __m256i v2_i32 = _mm256_cvtepi8_epi32(v2_i8);

                    __m256 v1_f = _mm256_cvtepi32_ps(v1_i32);
                    __m256 v2_f = _mm256_cvtepi32_ps(v2_i32);

                    v1_f = _mm256_mul_ps(v1_f, v_scale1);
                    v2_f = _mm256_mul_ps(v2_f, v_scale2);

                    __m256 diff = _mm256_sub_ps(v1_f, v2_f);
                    sum = _mm256_fmadd_ps(diff, diff, sum);
                }
                // Reduce AVX2 sum
                __m128 sum_lo = _mm256_castps256_ps128(sum);
                __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
                sum_lo = _mm_add_ps(sum_lo, sum_hi);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                res = _mm_cvtss_f32(sum_lo);
#elif defined(USE_SVE2)
                svbool_t pg = svwhilelt_b32(i, qty);
                svfloat32_t sum = svdup_f32(0.0f);
                svfloat32_t v_scale1 = svdup_f32(scale1);
                svfloat32_t v_scale2 = svdup_f32(scale2);

                while(svptest_any(svptrue_b32(), pg)) {
                    // Load int8 as int32
                    svint32_t v1_i32 = svld1sb_s32(pg, pVect1 + i);
                    svint32_t v2_i32 = svld1sb_s32(pg, pVect2 + i);

                    svfloat32_t v1_f = svcvt_f32_s32_x(pg, v1_i32);
                    svfloat32_t v2_f = svcvt_f32_s32_x(pg, v2_i32);

                    v1_f = svmul_f32_x(pg, v1_f, v_scale1);
                    v2_f = svmul_f32_x(pg, v2_f, v_scale2);

                    svfloat32_t diff = svsub_f32_x(pg, v1_f, v2_f);
                    sum = svmla_f32_x(pg, sum, diff, diff);

                    i += svcntw();
                    pg = svwhilelt_b32(i, qty);
                }
                res = svaddv_f32(svptrue_b32(), sum);
#elif defined(USE_NEON)
                // NEON implementation for L2Sqr
                // Uses the expansion: (a*s1 - b*s2)^2 = a^2*s1^2 + b^2*s2^2 - 2ab*s1*s2
                // This allows using integer dot products for the terms.

                float s1_sq = scale1 * scale1;
                float s2_sq = scale2 * scale2;
                float s1_s2_2 = 2.0f * scale1 * scale2;

                int32x4_t sum_sq1 = vdupq_n_s32(0);
                int32x4_t sum_sq2 = vdupq_n_s32(0);
                int32x4_t sum_prod = vdupq_n_s32(0);

#    if defined(__ARM_FEATURE_DOTPROD)
                size_t qty16 = qty / 16;
                for(; i < qty16 * 16; i += 16) {
                    int8x16_t v1 = vld1q_s8(pVect1 + i);
                    int8x16_t v2 = vld1q_s8(pVect2 + i);

                    sum_sq1 = vdotq_s32(sum_sq1, v1, v1);
                    sum_sq2 = vdotq_s32(sum_sq2, v2, v2);
                    sum_prod = vdotq_s32(sum_prod, v1, v2);
                }
#    else
                size_t qty16 = qty / 16;
                for(; i < qty16 * 16; i += 16) {
                    int8x16_t v1 = vld1q_s8(pVect1 + i);
                    int8x16_t v2 = vld1q_s8(pVect2 + i);

                    int16x8_t v1_lo = vmovl_s8(vget_low_s8(v1));
                    int16x8_t v1_hi = vmovl_s8(vget_high_s8(v1));
                    int16x8_t v2_lo = vmovl_s8(vget_low_s8(v2));
                    int16x8_t v2_hi = vmovl_s8(vget_high_s8(v2));

                    // v1^2
                    int32x4_t sq1_lo = vmull_s16(vget_low_s16(v1_lo), vget_low_s16(v1_lo));
                    int32x4_t sq1_hi = vmull_s16(vget_high_s16(v1_lo), vget_high_s16(v1_lo));
                    sum_sq1 = vaddq_s32(sum_sq1, vaddq_s32(sq1_lo, sq1_hi));
                    sq1_lo = vmull_s16(vget_low_s16(v1_hi), vget_low_s16(v1_hi));
                    sq1_hi = vmull_s16(vget_high_s16(v1_hi), vget_high_s16(v1_hi));
                    sum_sq1 = vaddq_s32(sum_sq1, vaddq_s32(sq1_lo, sq1_hi));

                    // v2^2
                    int32x4_t sq2_lo = vmull_s16(vget_low_s16(v2_lo), vget_low_s16(v2_lo));
                    int32x4_t sq2_hi = vmull_s16(vget_high_s16(v2_lo), vget_high_s16(v2_lo));
                    sum_sq2 = vaddq_s32(sum_sq2, vaddq_s32(sq2_lo, sq2_hi));
                    sq2_lo = vmull_s16(vget_low_s16(v2_hi), vget_low_s16(v2_hi));
                    sq2_hi = vmull_s16(vget_high_s16(v2_hi), vget_high_s16(v2_hi));
                    sum_sq2 = vaddq_s32(sum_sq2, vaddq_s32(sq2_lo, sq2_hi));

                    // v1*v2
                    int32x4_t prod_lo = vmull_s16(vget_low_s16(v1_lo), vget_low_s16(v2_lo));
                    int32x4_t prod_hi = vmull_s16(vget_high_s16(v1_lo), vget_high_s16(v2_lo));
                    sum_prod = vaddq_s32(sum_prod, vaddq_s32(prod_lo, prod_hi));
                    prod_lo = vmull_s16(vget_low_s16(v1_hi), vget_low_s16(v2_hi));
                    prod_hi = vmull_s16(vget_high_s16(v1_hi), vget_high_s16(v2_hi));
                    sum_prod = vaddq_s32(sum_prod, vaddq_s32(prod_lo, prod_hi));
                }
#    endif

                float dot1 = static_cast<float>(vaddvq_s32(sum_sq1));
                float dot2 = static_cast<float>(vaddvq_s32(sum_sq2));
                float dot_prod = static_cast<float>(vaddvq_s32(sum_prod));

                res = dot1 * s1_sq + dot2 * s2_sq - dot_prod * s1_s2_2;
#endif

                for(; i < qty; i++) {
                    float v1 = static_cast<float>(pVect1[i]) * scale1;
                    float v2 = static_cast<float>(pVect2[i]) * scale2;
                    float diff = v1 - v2;
                    res += diff * diff;
                }
                return res;
            }

            static float L2SqrSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return -L2Sqr(pVect1v, pVect2v, qty_ptr);
            }

            static float
            InnerProductSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = (const int8_t*)pVect1v;
                const int8_t* pVect2 = (const int8_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                size_t qty = params->dim;

                float scale1 = extract_inv_scale((const uint8_t*)pVect1, qty);
                float scale2 = extract_inv_scale((const uint8_t*)pVect2, qty);

                int32_t sum = 0;
                size_t i = 0;

#if defined(USE_AVX512)
                __m512i v_sum = _mm512_setzero_si512();
                for(; i + 32 <= qty; i += 32) {
                    __m256i v1_256 = _mm256_loadu_si256((const __m256i*)(pVect1 + i));
                    __m256i v2_256 = _mm256_loadu_si256((const __m256i*)(pVect2 + i));

                    // Sign extend to 16-bit
                    __m512i v1_512 = _mm512_cvtepi8_epi16(v1_256);
                    __m512i v2_512 = _mm512_cvtepi8_epi16(v2_256);

                    // Multiply and add adjacent pairs (produces 32-bit integers)
                    __m512i prod = _mm512_madd_epi16(v1_512, v2_512);

                    v_sum = _mm512_add_epi32(v_sum, prod);
                }
                sum = _mm512_reduce_add_epi32(v_sum);
#elif defined(USE_AVX2)
                __m256i v_sum = _mm256_setzero_si256();
                for(; i + 16 <= qty; i += 16) {
                    __m128i v1_128 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
                    __m128i v2_128 = _mm_loadu_si128((const __m128i*)(pVect2 + i));

                    __m256i v1_256 = _mm256_cvtepi8_epi16(v1_128);
                    __m256i v2_256 = _mm256_cvtepi8_epi16(v2_128);

                    __m256i prod = _mm256_madd_epi16(v1_256, v2_256);
                    v_sum = _mm256_add_epi32(v_sum, prod);
                }
                // Reduce AVX2 sum
                __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(v_sum),
                                                _mm256_extracti128_si256(v_sum, 1));
                sum_128 = _mm_hadd_epi32(sum_128, sum_128);
                sum_128 = _mm_hadd_epi32(sum_128, sum_128);
                sum = _mm_cvtsi128_si32(sum_128);
#elif defined(USE_SVE2)
                svbool_t pg8 = svwhilelt_b8(i, qty);
                svint32_t v_sum = svdup_s32(0);

                while(svptest_any(svptrue_b8(), pg8)) {
                    svint8_t v1 = svld1_s8(pg8, pVect1 + i);
                    svint8_t v2 = svld1_s8(pg8, pVect2 + i);
                    v_sum = svdot_s32(v_sum, v1, v2);

                    i += svcntb();
                    pg8 = svwhilelt_b8(i, qty);
                }
                sum = svaddv_s32(svptrue_b32(), v_sum);
#elif defined(USE_NEON)
                int32x4_t sum_vec0 = vdupq_n_s32(0);
                int32x4_t sum_vec1 = vdupq_n_s32(0);
                int32x4_t sum_vec2 = vdupq_n_s32(0);
                int32x4_t sum_vec3 = vdupq_n_s32(0);

// Use dot product instructions if available (ARMv8.2+)
#    if defined(__ARM_FEATURE_DOTPROD)
                size_t qty64 = qty / 64;
                for(; i < qty64 * 64; i += 64) {
                    int8x16_t v1_0 = vld1q_s8(pVect1 + i);
                    int8x16_t v2_0 = vld1q_s8(pVect2 + i);
                    int8x16_t v1_1 = vld1q_s8(pVect1 + i + 16);
                    int8x16_t v2_1 = vld1q_s8(pVect2 + i + 16);
                    int8x16_t v1_2 = vld1q_s8(pVect1 + i + 32);
                    int8x16_t v2_2 = vld1q_s8(pVect2 + i + 32);
                    int8x16_t v1_3 = vld1q_s8(pVect1 + i + 48);
                    int8x16_t v2_3 = vld1q_s8(pVect2 + i + 48);

                    sum_vec0 = vdotq_s32(sum_vec0, v1_0, v2_0);
                    sum_vec1 = vdotq_s32(sum_vec1, v1_1, v2_1);
                    sum_vec2 = vdotq_s32(sum_vec2, v1_2, v2_2);
                    sum_vec3 = vdotq_s32(sum_vec3, v1_3, v2_3);
                }

                // Handle remaining 16-element chunks
                for(; i < (qty / 16) * 16; i += 16) {
                    int8x16_t v1 = vld1q_s8(pVect1 + i);
                    int8x16_t v2 = vld1q_s8(pVect2 + i);
                    sum_vec0 = vdotq_s32(sum_vec0, v1, v2);
                }
#    else
                // Fallback for older NEON
                size_t qty16 = qty / 16;
                for(; i < qty16 * 16; i += 16) {
                    int8x16_t v1 = vld1q_s8(pVect1 + i);
                    int8x16_t v2 = vld1q_s8(pVect2 + i);

                    int16x8_t prod_low = vmull_s8(vget_low_s8(v1), vget_low_s8(v2));
                    int16x8_t prod_high = vmull_s8(vget_high_s8(v1), vget_high_s8(v2));

                    int32x4_t sum_low = vpaddlq_s16(prod_low);
                    int32x4_t sum_high = vpaddlq_s16(prod_high);

                    sum_vec0 = vaddq_s32(sum_vec0, vaddq_s32(sum_low, sum_high));
                }
#    endif

                sum_vec0 = vaddq_s32(sum_vec0, sum_vec1);
                sum_vec2 = vaddq_s32(sum_vec2, sum_vec3);
                sum_vec0 = vaddq_s32(sum_vec0, sum_vec2);
                sum = vaddvq_s32(sum_vec0);
#endif

                // Handle remaining
                for(; i < qty; i++) {
                    sum += static_cast<int32_t>(pVect1[i]) * static_cast<int32_t>(pVect2[i]);
                }

                return static_cast<float>(sum) * scale1 * scale2;
            }

            static float
            InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float CosineSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                // Vectors are normalized, so CosineSim is just InnerProductSim
                return InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float Cosine(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - CosineSim(pVect1v, pVect2v, qty_ptr);
            }

            // Direct quantization to INT8 - identity function for INT8 input
            static std::vector<uint8_t> quantize_to_int8_identity(const void* in, size_t dim) {
                size_t size = get_storage_size(dim);
                const uint8_t* ptr = static_cast<const uint8_t*>(in);
                return std::vector<uint8_t>(ptr, ptr + size);
            }

        }  // namespace int8d

        class Int8Quantizer : public ndd::quant::Quantizer {
        public:
            std::string name() const override { return "int8d"; }
            ndd::quant::QuantizationLevel level() const override {
                return ndd::quant::QuantizationLevel::INT8;
            }

            ndd::quant::QuantizerDispatch getDispatch() const override {
                ndd::quant::QuantizerDispatch d;
                d.dist_l2 = &int8d::L2Sqr;
                d.dist_ip = &int8d::InnerProduct;
                d.dist_cosine = &int8d::Cosine;
                d.sim_l2 = &int8d::L2SqrSim;
                d.sim_ip = &int8d::InnerProductSim;
                d.sim_cosine = &int8d::CosineSim;
                d.quantize = &int8d::quantize;
                d.dequantize = &int8d::dequantize;
                d.quantize_to_int8 = &int8d::quantize_to_int8_identity;
                d.get_storage_size = &int8d::get_storage_size;
                d.extract_inv_scale = &int8d::extract_inv_scale;
                return d;
            }
        };

        // Register INT8
        static ndd::quant::RegisterQuantizer reg_int8(ndd::quant::QuantizationLevel::INT8,
                                                      "int8d",
                                                      std::make_shared<Int8Quantizer>());

    }  // namespace quant
}  // namespace ndd
