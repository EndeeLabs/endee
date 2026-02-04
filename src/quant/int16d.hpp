#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "common.hpp"
#include "../hnsw/hnswlib.h"

namespace ndd {
    namespace quant {
        namespace int16d {

            constexpr float INT16_SCALE =
                    32767.0f;  // Max value for 16-bit signed integer quantization

            constexpr size_t get_storage_size(size_t dimension) {
                return dimension * sizeof(int16_t) + sizeof(float);
            }

            inline float extract_inv_scale(const uint8_t* buffer, size_t dimension) {
                return *reinterpret_cast<const float*>(buffer + dimension * sizeof(int16_t));
            }

            // Quantize FP32 vector to INT16 + scale, store in uint8_t buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int16_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;  // Avoid division by zero
                }
                float scale = INT16_SCALE / abs_max;

                // Quantize data
                int16_t* data_ptr = reinterpret_cast<int16_t*>(buffer.data());
                for(size_t i = 0; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int16_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP16_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int16_t)));
                *scale_ptr = abs_max / INT16_SCALE;

                return buffer;
            }

#if defined(USE_AVX512)
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int16_buffer_avx512(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT16_SCALE / abs_max;

                // SIMD quantization - using 24 ZMM registers (75% utilization)
                int16_t* data_ptr = reinterpret_cast<int16_t*>(buffer.data());
                const __m512 scale_vec = _mm512_set1_ps(scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 128) * 128;  // Process 128 floats -> 128 int16s (8x unrolling)

                // 8-way unrolled loop using maximum registers for better ILP
                for(; i < vec_size; i += 128) {
                    // Load 8 vectors (128 floats total)
                    __m512 vec0 = _mm512_loadu_ps(&input[i]);
                    __m512 vec1 = _mm512_loadu_ps(&input[i + 16]);
                    __m512 vec2 = _mm512_loadu_ps(&input[i + 32]);
                    __m512 vec3 = _mm512_loadu_ps(&input[i + 48]);
                    __m512 vec4 = _mm512_loadu_ps(&input[i + 64]);
                    __m512 vec5 = _mm512_loadu_ps(&input[i + 80]);
                    __m512 vec6 = _mm512_loadu_ps(&input[i + 96]);
                    __m512 vec7 = _mm512_loadu_ps(&input[i + 112]);

                    // Scale all 8 vectors in parallel
                    vec0 = _mm512_mul_ps(vec0, scale_vec);
                    vec1 = _mm512_mul_ps(vec1, scale_vec);
                    vec2 = _mm512_mul_ps(vec2, scale_vec);
                    vec3 = _mm512_mul_ps(vec3, scale_vec);
                    vec4 = _mm512_mul_ps(vec4, scale_vec);
                    vec5 = _mm512_mul_ps(vec5, scale_vec);
                    vec6 = _mm512_mul_ps(vec6, scale_vec);
                    vec7 = _mm512_mul_ps(vec7, scale_vec);

                    // Convert to int32 (with rounding)
                    __m512i int_vec0 = _mm512_cvtps_epi32(vec0);
                    __m512i int_vec1 = _mm512_cvtps_epi32(vec1);
                    __m512i int_vec2 = _mm512_cvtps_epi32(vec2);
                    __m512i int_vec3 = _mm512_cvtps_epi32(vec3);
                    __m512i int_vec4 = _mm512_cvtps_epi32(vec4);
                    __m512i int_vec5 = _mm512_cvtps_epi32(vec5);
                    __m512i int_vec6 = _mm512_cvtps_epi32(vec6);
                    __m512i int_vec7 = _mm512_cvtps_epi32(vec7);

                    // Pack to int16 (64 int32 -> 64 int16)
                    __m256i packed0 = _mm512_cvtepi32_epi16(int_vec0);
                    __m256i packed1 = _mm512_cvtepi32_epi16(int_vec1);
                    __m256i packed2 = _mm512_cvtepi32_epi16(int_vec2);
                    __m256i packed3 = _mm512_cvtepi32_epi16(int_vec3);
                    __m256i packed4 = _mm512_cvtepi32_epi16(int_vec4);
                    __m256i packed5 = _mm512_cvtepi32_epi16(int_vec5);
                    __m256i packed6 = _mm512_cvtepi32_epi16(int_vec6);
                    __m256i packed7 = _mm512_cvtepi32_epi16(int_vec7);

                    // Store all 8 packed vectors
                    _mm256_storeu_si256((__m256i*)&data_ptr[i], packed0);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 16], packed1);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 32], packed2);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 48], packed3);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 64], packed4);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 80], packed5);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 96], packed6);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i + 112], packed7);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (dimension / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m512 vec = _mm512_loadu_ps(&input[i]);
                    vec = _mm512_mul_ps(vec, scale_vec);

                    __m512i int_vec = _mm512_cvtps_epi32(vec);
                    __m256i packed = _mm512_cvtepi32_epi16(int_vec);
                    _mm256_storeu_si256((__m256i*)&data_ptr[i], packed);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int16_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP16_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int16_t)));
                *scale_ptr = abs_max / INT16_SCALE;

                return buffer;
            }
#endif

#if defined(USE_NEON)
            // NEON optimized quantization FP32 -> INT16 buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int16_buffer_neon(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT16_SCALE / abs_max;

                // SIMD quantization - using more registers for better parallelism
                int16_t* data_ptr = reinterpret_cast<int16_t*>(buffer.data());
                const float32x4_t scale_vec = vdupq_n_f32(scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 32) * 32;  // Process 32 floats -> 32 int16s (4x unrolling)

                // 4-way unrolled loop using 16 NEON registers for better ILP
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
                    int16x8_t packed0 = vcombine_s16(vqmovn_s32(int_vec0), vqmovn_s32(int_vec1));
                    int16x8_t packed1 = vcombine_s16(vqmovn_s32(int_vec2), vqmovn_s32(int_vec3));
                    int16x8_t packed2 = vcombine_s16(vqmovn_s32(int_vec4), vqmovn_s32(int_vec5));
                    int16x8_t packed3 = vcombine_s16(vqmovn_s32(int_vec6), vqmovn_s32(int_vec7));

                    // Store all 4 packed vectors
                    vst1q_s16(&data_ptr[i], packed0);
                    vst1q_s16(&data_ptr[i + 8], packed1);
                    vst1q_s16(&data_ptr[i + 16], packed2);
                    vst1q_s16(&data_ptr[i + 24], packed3);
                }

                // Handle remaining 8-element chunks
                size_t remaining_vec_size = (dimension / 8) * 8;
                for(; i < remaining_vec_size; i += 8) {
                    float32x4_t vec1 = vld1q_f32(&input[i]);
                    float32x4_t vec2 = vld1q_f32(&input[i + 4]);

                    vec1 = vmulq_f32(vec1, scale_vec);
                    vec2 = vmulq_f32(vec2, scale_vec);

                    int32x4_t int_vec1 = vcvtaq_s32_f32(vec1);
                    int32x4_t int_vec2 = vcvtaq_s32_f32(vec2);

                    int16x8_t packed = vcombine_s16(vqmovn_s32(int_vec1), vqmovn_s32(int_vec2));
                    vst1q_s16(&data_ptr[i], packed);
                }

                // Handle remaining elements
                for(; i < dimension; ++i) {
                    float scaled = input[i] * scale;
                    data_ptr[i] = static_cast<int16_t>(std::round(scaled));
                }

                // Store INVERSE scale for dequantization (abs_max/FP16_SCALE)
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int16_t)));
                *scale_ptr = abs_max / INT16_SCALE;

                return buffer;
            }
#endif

#if defined(USE_SVE2)
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int16_buffer_sve(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                // Find scale factor
                float abs_max = math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = INT16_SCALE / abs_max;

                int16_t* data_ptr = reinterpret_cast<int16_t*>(buffer.data());

                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dimension);

                while(svptest_any(svptrue_b32(), pg)) {
                    svfloat32_t vec = svld1_f32(pg, &input[i]);
                    vec = svmul_f32_x(pg, vec, svdup_f32(scale));

                    // Convert to int32 (with rounding)
                    svint32_t int_vec = svcvt_s32_f32_x(pg, vec);

                    // Store lower 16 bits of each 32-bit element to contiguous memory
                    svst1h_s32(pg, &data_ptr[i], int_vec);

                    i += svcntw();
                    pg = svwhilelt_b32(i, dimension);
                }

                // Store INVERSE scale for dequantization
                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int16_t)));
                *scale_ptr = abs_max / INT16_SCALE;

                return buffer;
            }
#endif

            // Auto-select best quantization implementation for INT16 -> uint8_t buffer
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int16_buffer_auto(const std::vector<float>& input) {
#if defined(USE_AVX512)
                return quantize_vector_fp32_to_int16_buffer_avx512(input);
#elif defined(USE_SVE2)
                return quantize_vector_fp32_to_int16_buffer_sve(input);
#elif defined(USE_NEON)
                return quantize_vector_fp32_to_int16_buffer_neon(input);
#else
                return quantize_vector_fp32_to_int16_buffer(input);
#endif
            }

#if defined(USE_AVX512)
            // AVX512 optimized dequantization INT16 buffer -> FP32 vector
            inline std::vector<float> dequantize_int16_buffer_to_fp32_avx512(const uint8_t* buffer,
                                                                             size_t dimension) {
                std::vector<float> output(dimension);
                const int16_t* data_ptr = reinterpret_cast<const int16_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 32) * 32;  // Process 32 int16s -> 32 floats (2x unrolling)

                // 2-way unrolled loop using multiple ZMM registers
                for(; i < vec_size; i += 32) {
                    // Load 32 int16 values (2 x 256-bit loads)
                    __m256i int16_vec0 = _mm256_loadu_si256((__m256i*)&data_ptr[i]);
                    __m256i int16_vec1 = _mm256_loadu_si256((__m256i*)&data_ptr[i + 16]);

                    // Convert to int32 (sign extend)
                    __m512i int32_vec0 = _mm512_cvtepi16_epi32(int16_vec0);
                    __m512i int32_vec1 = _mm512_cvtepi16_epi32(int16_vec1);

                    // Convert to float
                    __m512 float_vec0 = _mm512_cvtepi32_ps(int32_vec0);
                    __m512 float_vec1 = _mm512_cvtepi32_ps(int32_vec1);

                    // Apply inverse scale
                    float_vec0 = _mm512_mul_ps(float_vec0, inv_scale_vec);
                    float_vec1 = _mm512_mul_ps(float_vec1, inv_scale_vec);

                    // Store results
                    _mm512_storeu_ps(&output[i], float_vec0);
                    _mm512_storeu_ps(&output[i + 16], float_vec1);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (dimension / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m256i int16_vec = _mm256_loadu_si256((__m256i*)&data_ptr[i]);
                    __m512i int32_vec = _mm512_cvtepi16_epi32(int16_vec);
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
            inline std::vector<float> dequantize_int16_buffer_to_fp32_neon(const uint8_t* buffer,
                                                                           size_t dimension) {
                std::vector<float> output(dimension);
                const int16_t* data_ptr = reinterpret_cast<const int16_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                const float32x4_t inv_scale_vec = vdupq_n_f32(1.0f / scale);

                size_t i = 0;
                size_t vec_size =
                        (dimension / 16) * 16;  // Process 16 int16s -> 16 floats (4x unrolling)

                for(; i < vec_size; i += 16) {
                    // Load 16 int16 values (2 x 128-bit loads)
                    int16x8_t int16_vec0 = vld1q_s16(&data_ptr[i]);
                    int16x8_t int16_vec1 = vld1q_s16(&data_ptr[i + 8]);

                    // Convert to int32 (4 parts)
                    int32x4_t int32_0 = vmovl_s16(vget_low_s16(int16_vec0));
                    int32x4_t int32_1 = vmovl_s16(vget_high_s16(int16_vec0));
                    int32x4_t int32_2 = vmovl_s16(vget_low_s16(int16_vec1));
                    int32x4_t int32_3 = vmovl_s16(vget_high_s16(int16_vec1));

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
            inline std::vector<float> dequantize_int16_buffer_to_fp32_sve(const uint8_t* buffer,
                                                                          size_t dimension) {
                std::vector<float> output(dimension);
                const int16_t* data_ptr = reinterpret_cast<const int16_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dimension);

                while(svptest_any(svptrue_b32(), pg)) {
                    // Load 16-bit integers into 32-bit vector elements (sign-extended)
                    svint32_t int_vec = svld1sh_s32(pg, &data_ptr[i]);

                    // Convert to float
                    svfloat32_t float_vec = svcvt_f32_s32_x(pg, int_vec);

                    // Apply inverse scale (division by scale)
                    float_vec = svdiv_f32_x(pg, float_vec, svdup_f32(scale));

                    // Store results
                    svst1_f32(pg, &output[i], float_vec);

                    i += svcntw();
                    pg = svwhilelt_b32(i, dimension);
                }

                return output;
            }
#endif

            // Auto-select best dequantization implementation for INT16 buffer -> FP32 vector
            inline std::vector<float> dequantize_int16_buffer_to_fp32(const uint8_t* buffer,
                                                                      size_t dimension) {
#if defined(USE_AVX512)
                return dequantize_int16_buffer_to_fp32_avx512(buffer, dimension);
#elif defined(USE_SVE2)
                return dequantize_int16_buffer_to_fp32_sve(buffer, dimension);
#elif defined(USE_NEON)
                return dequantize_int16_buffer_to_fp32_neon(buffer, dimension);
#else
                // Scalar fallback
                std::vector<float> output(dimension);
                const int16_t* data_ptr = reinterpret_cast<const int16_t*>(buffer);
                float scale = extract_inv_scale(buffer, dimension);

                for(size_t i = 0; i < dimension; ++i) {
                    output[i] = static_cast<float>(data_ptr[i]) / scale;
                }
                return output;
#endif
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int16_buffer_auto(input);
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                return dequantize_int16_buffer_to_fp32(in, dim);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int16_t* pVect1 = (const int16_t*)pVect1v;
                const int16_t* pVect2 = (const int16_t*)pVect2v;
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
                    __m256i v1_i16 = _mm256_loadu_si256((const __m256i*)(pVect1 + i));
                    __m256i v2_i16 = _mm256_loadu_si256((const __m256i*)(pVect2 + i));

                    __m512i v1_i32 = _mm512_cvtepi16_epi32(v1_i16);
                    __m512i v2_i32 = _mm512_cvtepi16_epi32(v2_i16);

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
                    __m128i v1_i16 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
                    __m128i v2_i16 = _mm_loadu_si128((const __m128i*)(pVect2 + i));

                    __m256i v1_i32 = _mm256_cvtepi16_epi32(v1_i16);
                    __m256i v2_i32 = _mm256_cvtepi16_epi32(v2_i16);

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
#elif defined(USE_NEON)
                float32x4_t sum0 = vdupq_n_f32(0);
                float32x4_t sum1 = vdupq_n_f32(0);
                float32x4_t sum2 = vdupq_n_f32(0);
                float32x4_t sum3 = vdupq_n_f32(0);

                float32x4_t v_scale1 = vdupq_n_f32(scale1);
                float32x4_t v_scale2 = vdupq_n_f32(scale2);

                size_t qty16 = qty / 16;
                for(; i < qty16 * 16; i += 16) {
                    int16x8_t v1_0 = vld1q_s16(pVect1 + i);
                    int16x8_t v2_0 = vld1q_s16(pVect2 + i);
                    int16x8_t v1_1 = vld1q_s16(pVect1 + i + 8);
                    int16x8_t v2_1 = vld1q_s16(pVect2 + i + 8);

                    // Low parts
                    float32x4_t f1_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1_0)));
                    float32x4_t f2_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v2_0)));
                    f1_0 = vmulq_f32(f1_0, v_scale1);
                    f2_0 = vmulq_f32(f2_0, v_scale2);
                    float32x4_t diff0 = vsubq_f32(f1_0, f2_0);
                    sum0 = vmlaq_f32(sum0, diff0, diff0);

                    // High parts
                    float32x4_t f1_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v1_0)));
                    float32x4_t f2_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v2_0)));
                    f1_1 = vmulq_f32(f1_1, v_scale1);
                    f2_1 = vmulq_f32(f2_1, v_scale2);
                    float32x4_t diff1 = vsubq_f32(f1_1, f2_1);
                    sum1 = vmlaq_f32(sum1, diff1, diff1);

                    // Next 8
                    float32x4_t f1_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1_1)));
                    float32x4_t f2_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v2_1)));
                    f1_2 = vmulq_f32(f1_2, v_scale1);
                    f2_2 = vmulq_f32(f2_2, v_scale2);
                    float32x4_t diff2 = vsubq_f32(f1_2, f2_2);
                    sum2 = vmlaq_f32(sum2, diff2, diff2);

                    float32x4_t f1_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v1_1)));
                    float32x4_t f2_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v2_1)));
                    f1_3 = vmulq_f32(f1_3, v_scale1);
                    f2_3 = vmulq_f32(f2_3, v_scale2);
                    float32x4_t diff3 = vsubq_f32(f1_3, f2_3);
                    sum3 = vmlaq_f32(sum3, diff3, diff3);
                }

                sum0 = vaddq_f32(sum0, sum1);
                sum2 = vaddq_f32(sum2, sum3);
                sum0 = vaddq_f32(sum0, sum2);
                res = vaddvq_f32(sum0);
#elif defined(USE_SVE2)
                svfloat32_t sum0 = svdup_f32(0.0f);
                svfloat32_t sum1 = svdup_f32(0.0f);
                svfloat32_t sum2 = svdup_f32(0.0f);
                svfloat32_t sum3 = svdup_f32(0.0f);

                uint64_t num_elements = svcnth();
                size_t unroll_stride = num_elements * 2;
                svbool_t pg_all = svptrue_b16();

                // Main unrolled loop
                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svint16_t v1_0 = svld1_s16(pg_all, pVect1 + i);
                    svint16_t v2_0 = svld1_s16(pg_all, pVect2 + i);

                    svint32_t v1_0_lo = svunpklo_s32(v1_0);
                    svint32_t v1_0_hi = svunpkhi_s32(v1_0);
                    svint32_t v2_0_lo = svunpklo_s32(v2_0);
                    svint32_t v2_0_hi = svunpkhi_s32(v2_0);

                    svfloat32_t f1_0_lo = svcvt_f32_s32_x(svptrue_b32(), v1_0_lo);
                    svfloat32_t f1_0_hi = svcvt_f32_s32_x(svptrue_b32(), v1_0_hi);
                    svfloat32_t f2_0_lo = svcvt_f32_s32_x(svptrue_b32(), v2_0_lo);
                    svfloat32_t f2_0_hi = svcvt_f32_s32_x(svptrue_b32(), v2_0_hi);

                    f1_0_lo = svmul_n_f32_x(svptrue_b32(), f1_0_lo, scale1);
                    f1_0_hi = svmul_n_f32_x(svptrue_b32(), f1_0_hi, scale1);
                    f2_0_lo = svmul_n_f32_x(svptrue_b32(), f2_0_lo, scale2);
                    f2_0_hi = svmul_n_f32_x(svptrue_b32(), f2_0_hi, scale2);

                    svfloat32_t diff_0_lo = svsub_f32_x(svptrue_b32(), f1_0_lo, f2_0_lo);
                    svfloat32_t diff_0_hi = svsub_f32_x(svptrue_b32(), f1_0_hi, f2_0_hi);

                    sum0 = svmla_f32_x(svptrue_b32(), sum0, diff_0_lo, diff_0_lo);
                    sum1 = svmla_f32_x(svptrue_b32(), sum1, diff_0_hi, diff_0_hi);

                    svint16_t v1_1 = svld1_s16(pg_all, pVect1 + i + num_elements);
                    svint16_t v2_1 = svld1_s16(pg_all, pVect2 + i + num_elements);

                    svint32_t v1_1_lo = svunpklo_s32(v1_1);
                    svint32_t v1_1_hi = svunpkhi_s32(v1_1);
                    svint32_t v2_1_lo = svunpklo_s32(v2_1);
                    svint32_t v2_1_hi = svunpkhi_s32(v2_1);

                    svfloat32_t f1_1_lo = svcvt_f32_s32_x(svptrue_b32(), v1_1_lo);
                    svfloat32_t f1_1_hi = svcvt_f32_s32_x(svptrue_b32(), v1_1_hi);
                    svfloat32_t f2_1_lo = svcvt_f32_s32_x(svptrue_b32(), v2_1_lo);
                    svfloat32_t f2_1_hi = svcvt_f32_s32_x(svptrue_b32(), v2_1_hi);

                    f1_1_lo = svmul_n_f32_x(svptrue_b32(), f1_1_lo, scale1);
                    f1_1_hi = svmul_n_f32_x(svptrue_b32(), f1_1_hi, scale1);
                    f2_1_lo = svmul_n_f32_x(svptrue_b32(), f2_1_lo, scale2);
                    f2_1_hi = svmul_n_f32_x(svptrue_b32(), f2_1_hi, scale2);

                    svfloat32_t diff_1_lo = svsub_f32_x(svptrue_b32(), f1_1_lo, f2_1_lo);
                    svfloat32_t diff_1_hi = svsub_f32_x(svptrue_b32(), f1_1_hi, f2_1_hi);

                    sum2 = svmla_f32_x(svptrue_b32(), sum2, diff_1_lo, diff_1_lo);
                    sum3 = svmla_f32_x(svptrue_b32(), sum3, diff_1_hi, diff_1_hi);
                }

                svfloat32_t sum_vec = svadd_f32_x(svptrue_b32(), sum0, sum1);
                sum_vec = svadd_f32_x(svptrue_b32(), sum_vec, sum2);
                sum_vec = svadd_f32_x(svptrue_b32(), sum_vec, sum3);

                svbool_t pg = svwhilelt_b32(i, qty);

                while(svptest_any(svptrue_b32(), pg)) {
                    svint32_t v1_i32 = svld1sh_s32(pg, pVect1 + i);
                    svint32_t v2_i32 = svld1sh_s32(pg, pVect2 + i);

                    svfloat32_t v1_f = svcvt_f32_s32_x(pg, v1_i32);
                    svfloat32_t v2_f = svcvt_f32_s32_x(pg, v2_i32);

                    v1_f = svmul_n_f32_x(pg, v1_f, scale1);
                    v2_f = svmul_n_f32_x(pg, v2_f, scale2);

                    svfloat32_t diff = svsub_f32_x(pg, v1_f, v2_f);
                    sum_vec = svmla_f32_x(pg, sum_vec, diff, diff);

                    i += svcntw();
                    pg = svwhilelt_b32(i, qty);
                }
                res = svaddv_f32(svptrue_b32(), sum_vec);
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
                const int16_t* pVect1 = (const int16_t*)pVect1v;
                const int16_t* pVect2 = (const int16_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                size_t qty = params->dim;

                float scale1 = extract_inv_scale((const uint8_t*)pVect1, qty);
                float scale2 = extract_inv_scale((const uint8_t*)pVect2, qty);

                int64_t sum = 0;
                size_t i = 0;

#if defined(USE_AVX512)
                __m512i sum_vec = _mm512_setzero_si512();

                for(; i + 32 <= qty; i += 32) {
                    __m512i v1 = _mm512_loadu_si512((const __m512i*)(pVect1 + i));
                    __m512i v2 = _mm512_loadu_si512((const __m512i*)(pVect2 + i));

                    // Multiply and add adjacent pairs -> 16 x 32-bit integers
                    __m512i prod = _mm512_madd_epi16(v1, v2);

                    // Extend to 64-bit and accumulate
                    __m512i prod_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prod));
                    __m512i prod_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(prod, 1));

                    sum_vec = _mm512_add_epi64(sum_vec, prod_lo);
                    sum_vec = _mm512_add_epi64(sum_vec, prod_hi);
                }
                sum = _mm512_reduce_add_epi64(sum_vec);
#elif defined(USE_AVX2)
                __m256i sum_vec = _mm256_setzero_si256();

                for(; i + 16 <= qty; i += 16) {
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(pVect1 + i));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(pVect2 + i));

                    // 8 x 32-bit integers
                    __m256i prod = _mm256_madd_epi16(v1, v2);

                    // Extend to 64-bit
                    __m256i prod_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod));
                    __m256i prod_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1));

                    sum_vec = _mm256_add_epi64(sum_vec, prod_lo);
                    sum_vec = _mm256_add_epi64(sum_vec, prod_hi);
                }
                // Reduce AVX2 sum (4 x 64-bit)
                __m128i sum_128 = _mm_add_epi64(_mm256_castsi256_si128(sum_vec),
                                                _mm256_extracti128_si256(sum_vec, 1));
                __m128i high64 = _mm_unpackhi_epi64(sum_128, sum_128);
                sum_128 = _mm_add_epi64(sum_128, high64);
                sum = _mm_cvtsi128_si64(sum_128);
#elif defined(USE_NEON)
                int64x2_t sum_vec0 = vdupq_n_s64(0);
                int64x2_t sum_vec1 = vdupq_n_s64(0);

                size_t qty16 = qty / 16;
                for(; i < qty16 * 16; i += 16) {
                    int16x8_t v1_0 = vld1q_s16(pVect1 + i);
                    int16x8_t v2_0 = vld1q_s16(pVect2 + i);
                    int16x8_t v1_1 = vld1q_s16(pVect1 + i + 8);
                    int16x8_t v2_1 = vld1q_s16(pVect2 + i + 8);

                    // Multiply -> 32-bit
                    int32x4_t prod0_lo = vmull_s16(vget_low_s16(v1_0), vget_low_s16(v2_0));
                    int32x4_t prod0_hi = vmull_s16(vget_high_s16(v1_0), vget_high_s16(v2_0));
                    int32x4_t prod1_lo = vmull_s16(vget_low_s16(v1_1), vget_low_s16(v2_1));
                    int32x4_t prod1_hi = vmull_s16(vget_high_s16(v1_1), vget_high_s16(v2_1));

                    // Accumulate into 64-bit
                    sum_vec0 = vpadalq_s32(sum_vec0, prod0_lo);
                    sum_vec0 = vpadalq_s32(sum_vec0, prod0_hi);
                    sum_vec1 = vpadalq_s32(sum_vec1, prod1_lo);
                    sum_vec1 = vpadalq_s32(sum_vec1, prod1_hi);
                }

                sum_vec0 = vaddq_s64(sum_vec0, sum_vec1);
                sum = vgetq_lane_s64(sum_vec0, 0) + vgetq_lane_s64(sum_vec0, 1);
#elif defined(USE_SVE2)
                uint64_t num_elements = svcnth();
                size_t unroll_stride = num_elements * 4;
                svbool_t pg_all = svptrue_b16();
                svbool_t pg_64 = svptrue_b64();
                svint32_t zero_s32 = svdup_s32(0);

                svint64_t sum0 = svdup_s64(0);
                svint64_t sum1 = svdup_s64(0);
                svint64_t sum2 = svdup_s64(0);
                svint64_t sum3 = svdup_s64(0);

                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svint16_t v1_0 = svld1_s16(pg_all, pVect1 + i);
                    svint16_t v2_0 = svld1_s16(pg_all, pVect2 + i);
                    svint32_t p_lo_0 = svmlalb_s32(zero_s32, v1_0, v2_0);
                    svint32_t p_hi_0 = svmlalt_s32(zero_s32, v1_0, v2_0);
                    sum0 = svadd_s64_x(pg_64, sum0, svaddlb_s64(p_lo_0, zero_s32));
                    sum1 = svadd_s64_x(pg_64, sum1, svaddlt_s64(p_lo_0, zero_s32));
                    sum2 = svadd_s64_x(pg_64, sum2, svaddlb_s64(p_hi_0, zero_s32));
                    sum3 = svadd_s64_x(pg_64, sum3, svaddlt_s64(p_hi_0, zero_s32));

                    svint16_t v1_1 = svld1_s16(pg_all, pVect1 + i + num_elements);
                    svint16_t v2_1 = svld1_s16(pg_all, pVect2 + i + num_elements);
                    svint32_t p_lo_1 = svmlalb_s32(zero_s32, v1_1, v2_1);
                    svint32_t p_hi_1 = svmlalt_s32(zero_s32, v1_1, v2_1);
                    sum0 = svadd_s64_x(pg_64, sum0, svaddlb_s64(p_lo_1, zero_s32));
                    sum1 = svadd_s64_x(pg_64, sum1, svaddlt_s64(p_lo_1, zero_s32));
                    sum2 = svadd_s64_x(pg_64, sum2, svaddlb_s64(p_hi_1, zero_s32));
                    sum3 = svadd_s64_x(pg_64, sum3, svaddlt_s64(p_hi_1, zero_s32));

                    svint16_t v1_2 = svld1_s16(pg_all, pVect1 + i + 2 * num_elements);
                    svint16_t v2_2 = svld1_s16(pg_all, pVect2 + i + 2 * num_elements);
                    svint32_t p_lo_2 = svmlalb_s32(zero_s32, v1_2, v2_2);
                    svint32_t p_hi_2 = svmlalt_s32(zero_s32, v1_2, v2_2);
                    sum0 = svadd_s64_x(pg_64, sum0, svaddlb_s64(p_lo_2, zero_s32));
                    sum1 = svadd_s64_x(pg_64, sum1, svaddlt_s64(p_lo_2, zero_s32));
                    sum2 = svadd_s64_x(pg_64, sum2, svaddlb_s64(p_hi_2, zero_s32));
                    sum3 = svadd_s64_x(pg_64, sum3, svaddlt_s64(p_hi_2, zero_s32));

                    svint16_t v1_3 = svld1_s16(pg_all, pVect1 + i + 3 * num_elements);
                    svint16_t v2_3 = svld1_s16(pg_all, pVect2 + i + 3 * num_elements);
                    svint32_t p_lo_3 = svmlalb_s32(zero_s32, v1_3, v2_3);
                    svint32_t p_hi_3 = svmlalt_s32(zero_s32, v1_3, v2_3);
                    sum0 = svadd_s64_x(pg_64, sum0, svaddlb_s64(p_lo_3, zero_s32));
                    sum1 = svadd_s64_x(pg_64, sum1, svaddlt_s64(p_lo_3, zero_s32));
                    sum2 = svadd_s64_x(pg_64, sum2, svaddlb_s64(p_hi_3, zero_s32));
                    sum3 = svadd_s64_x(pg_64, sum3, svaddlt_s64(p_hi_3, zero_s32));
                }

                svint64_t sum_vec = svadd_s64_x(svptrue_b64(), sum0, sum1);
                sum_vec = svadd_s64_x(svptrue_b64(), sum_vec, sum2);
                sum_vec = svadd_s64_x(svptrue_b64(), sum_vec, sum3);

                svbool_t pg = svwhilelt_b64(i, qty);
                while(svptest_any(svptrue_b64(), pg)) {
                    svint64_t v1 = svld1sh_s64(pg, pVect1 + i);
                    svint64_t v2 = svld1sh_s64(pg, pVect2 + i);
                    svint64_t prod = svmul_s64_x(pg, v1, v2);
                    sum_vec = svadd_s64_x(pg, sum_vec, prod);
                    i += svcntd();
                    pg = svwhilelt_b64(i, qty);
                }
                sum = svaddv_s64(svptrue_b64(), sum_vec);
#endif

                for(; i < qty; i++) {
                    sum += static_cast<int64_t>(pVect1[i]) * static_cast<int64_t>(pVect2[i]);
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

            // Direct Int16 -> Int8 quantization
            static std::vector<uint8_t> quantize_to_int8(const void* in, size_t dim) {
                const int16_t* in_data = static_cast<const int16_t*>(in);
                // Get scale
                float inv_scale = extract_inv_scale(reinterpret_cast<const uint8_t*>(in), dim);

                // Target: value = stored_8 * new_inv_scale
                // We set: stored_8 = stored_16 / 256
                // new_inv_scale = inv_scale * 256.0f
                float new_inv_scale = inv_scale * 256.0f;

                size_t out_size = dim * sizeof(int8_t) + sizeof(float);
                std::vector<uint8_t> out_vec(out_size);
                int8_t* out_data = reinterpret_cast<int8_t*>(out_vec.data());

                size_t i = 0;

#if defined(USE_AVX512)
                for(; i + 64 <= dim; i += 64) {
                    __m512i v1 = _mm512_loadu_si512((const __m512i*)(in_data + i));
                    __m512i v2 = _mm512_loadu_si512((const __m512i*)(in_data + i + 32));
                    v1 = _mm512_srai_epi16(v1, 8);
                    v2 = _mm512_srai_epi16(v2, 8);
                    _mm256_storeu_si256((__m256i*)(out_data + i), _mm512_cvtepi16_epi8(v1));
                    _mm256_storeu_si256((__m256i*)(out_data + i + 32), _mm512_cvtepi16_epi8(v2));
                }
#elif defined(USE_AVX2)
                for(; i + 32 <= dim; i += 32) {
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(in_data + i));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(in_data + i + 16));
                    v1 = _mm256_srai_epi16(v1, 8);
                    v2 = _mm256_srai_epi16(v2, 8);
                    _mm_storeu_si128((__m128i*)(out_data + i),
                                     _mm_packs_epi16(_mm256_castsi256_si128(v1),
                                                     _mm256_extracti128_si256(v1, 1)));
                    _mm_storeu_si128((__m128i*)(out_data + i + 16),
                                     _mm_packs_epi16(_mm256_castsi256_si128(v2),
                                                     _mm256_extracti128_si256(v2, 1)));
                }
#elif defined(USE_NEON)
                for(; i + 32 <= dim; i += 32) {
                    int8x16_t p1 = vcombine_s8(vqshrn_n_s16(vld1q_s16(in_data + i), 8),
                                               vqshrn_n_s16(vld1q_s16(in_data + i + 8), 8));
                    int8x16_t p2 = vcombine_s8(vqshrn_n_s16(vld1q_s16(in_data + i + 16), 8),
                                               vqshrn_n_s16(vld1q_s16(in_data + i + 24), 8));
                    vst1q_s8(out_data + i, p1);
                    vst1q_s8(out_data + i + 16, p2);
                }
#endif
                for(; i < dim; ++i) {
                    out_data[i] = static_cast<int8_t>(in_data[i] >> 8);
                }
                std::memcpy(out_data + dim, &new_inv_scale, sizeof(float));
                return out_vec;
            }

        }  // namespace int16d

        class Int16Quantizer : public Quantizer {
        public:
            std::string name() const override { return "int16d"; }
            QuantizationLevel level() const override { return QuantizationLevel::INT16; }

            QuantizerDispatch getDispatch() const override {
                QuantizerDispatch d;
                d.dist_l2 = &int16d::L2Sqr;
                d.dist_ip = &int16d::InnerProduct;
                d.dist_cosine = &int16d::Cosine;
                d.sim_l2 = &int16d::L2SqrSim;
                d.sim_ip = &int16d::InnerProductSim;
                d.sim_cosine = &int16d::CosineSim;
                d.quantize = &int16d::quantize;
                d.dequantize = &int16d::dequantize;
                d.quantize_to_int8 = &int16d::quantize_to_int8;
                d.get_storage_size = &int16d::get_storage_size;
                d.extract_inv_scale = &int16d::extract_inv_scale;
                return d;
            }
        };

        // Register INT16
        static RegisterQuantizer
                reg_int16(QuantizationLevel::INT16, "int16d", std::make_shared<Int16Quantizer>());

    }  // namespace quant
}  // namespace ndd
