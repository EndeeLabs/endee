#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include "../hnsw/hnswlib.h"
#include "common.hpp"
#include "int8d.hpp"

namespace ndd {
    namespace quant {
        namespace float16 {

            constexpr size_t get_storage_size(size_t dimension) {
                return dimension * sizeof(uint16_t);
            }

            inline float extract_inv_scale(const uint8_t* in, size_t dim) {
                return 1.0f;
            }

            // Base scalar implementation for FP16->FP32
            inline float fp16_to_fp32(uint16_t h) {
                unsigned sign = ((h >> 15) & 1);
                unsigned exponent = ((h >> 10) & 0x1f);
                unsigned mantissa = ((h & 0x3ff) << 13);

                if(exponent == 0) {
                    if(mantissa == 0) {
                        return sign ? -0.0f : 0.0f;
                    } else {
                        while(!(mantissa & 0x800000)) {
                            mantissa <<= 1;
                            exponent -= 1;
                        }
                        exponent += 1;
                        mantissa &= ~0x800000;
                    }
                } else if(exponent == 31) {
                    if(mantissa == 0) {
                        return sign ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::infinity();
                    } else {
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                }

                exponent = exponent + (127 - 15);
                exponent = exponent << 23;

                unsigned int bits = (sign << 31) | exponent | mantissa;
                return *reinterpret_cast<float*>(&bits);
            }
            // Base scalar implementation for FP32->FP16
            inline uint16_t fp32_to_fp16(float f) {
                uint32_t x = *reinterpret_cast<uint32_t*>(&f);
                uint32_t sign = (x >> 31) & 0x1;
                uint32_t exp = (x >> 23) & 0xff;
                uint32_t mantissa = x & 0x7fffff;

                if(exp == 0xff) {  // Handle inf/NaN
                    if(mantissa == 0) {
                        return sign << 15 | 0x7c00;  // inf
                    }
                    return sign << 15 | 0x7c00 | (mantissa >> 13);  // NaN
                }

                if(exp <= 112) {  // Handle zero/denormal
                    return sign << 15;
                }

                if(exp >= 143) {  // Handle overflow
                    return sign << 15 | 0x7c00;
                }

                exp = exp - 127 + 15;
                mantissa = mantissa >> 13;

                return (sign << 15) | (exp << 10) | mantissa;
            }

#if defined(USE_NEON)
            // NEON optimized vector conversion FP16->FP32
            inline std::vector<float>
            convert_vector_f16_f32_neon(const std::vector<uint16_t>& input) {
                std::vector<float> output;
                output.resize(input.size());

                size_t i = 0;
                size_t vec_size =
                        (input.size() / 16) * 16;  // Process 16 fp16s -> 16 fp32s (4x unrolling)

                // 4-way unrolled loop using 8 NEON registers for better ILP
                for(; i < vec_size; i += 16) {
                    // Load 4 vectors of 4 FP16 values each (16 total)
                    float16x4_t in0 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i]));
                    float16x4_t in1 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 4]));
                    float16x4_t in2 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 8]));
                    float16x4_t in3 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 12]));

                    // Convert all 4 vectors to FP32 in parallel
                    float32x4_t out0 = vcvt_f32_f16(in0);
                    float32x4_t out1 = vcvt_f32_f16(in1);
                    float32x4_t out2 = vcvt_f32_f16(in2);
                    float32x4_t out3 = vcvt_f32_f16(in3);

                    // Store all 4 vectors
                    vst1q_f32(&output[i], out0);
                    vst1q_f32(&output[i + 4], out1);
                    vst1q_f32(&output[i + 8], out2);
                    vst1q_f32(&output[i + 12], out3);
                }

                // Handle remaining 4-element chunks
                size_t remaining_vec_size = (input.size() / 4) * 4;
                for(; i < remaining_vec_size; i += 4) {
                    float16x4_t in = vld1_f16(reinterpret_cast<const __fp16*>(&input[i]));
                    float32x4_t out = vcvt_f32_f16(in);
                    vst1q_f32(&output[i], out);
                }

                // Handle remaining elements
                for(; i < input.size(); i++) {
                    output[i] = fp16_to_fp32(input[i]);
                }

                return output;
            }

            // NEON optimized vector conversion FP32->FP16
            inline std::vector<uint16_t>
            convert_vector_f32_f16_neon(const std::vector<float>& input) {
                std::vector<uint16_t> output;
                output.resize(input.size());

                size_t i = 0;
                size_t vec_size =
                        (input.size() / 16) * 16;  // Process 16 fp32s -> 16 fp16s (4x unrolling)

                // 4-way unrolled loop using 8 NEON registers for better ILP
                for(; i < vec_size; i += 16) {
                    // Load 4 vectors of 4 FP32 values each (16 total)
                    float32x4_t in0 = vld1q_f32(&input[i]);
                    float32x4_t in1 = vld1q_f32(&input[i + 4]);
                    float32x4_t in2 = vld1q_f32(&input[i + 8]);
                    float32x4_t in3 = vld1q_f32(&input[i + 12]);

                    // Convert all 4 vectors to FP16 in parallel
                    float16x4_t out0 = vcvt_f16_f32(in0);
                    float16x4_t out1 = vcvt_f16_f32(in1);
                    float16x4_t out2 = vcvt_f16_f32(in2);
                    float16x4_t out3 = vcvt_f16_f32(in3);

                    // Store all 4 vectors
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i]), out0);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 4]), out1);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 8]), out2);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 12]), out3);
                }

                // Handle remaining 4-element chunks
                size_t remaining_vec_size = (input.size() / 4) * 4;
                for(; i < remaining_vec_size; i += 4) {
                    float32x4_t in = vld1q_f32(&input[i]);
                    float16x4_t out = vcvt_f16_f32(in);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i]), out);
                }

                // Handle remaining elements
                for(; i < input.size(); i++) {
                    output[i] = fp32_to_fp16(input[i]);
                }

                return output;
            }
#endif

#if defined(USE_AVX512)
            // AVX512 optimized vector conversion FP16->FP32
            inline std::vector<float>
            convert_vector_f16_f32_avx512(const std::vector<uint16_t>& input) {
                std::vector<float> output;
                output.resize(input.size());

                size_t i = 0;
                size_t vec_size =
                        (input.size() / 64) * 64;  // Process 64 fp16s -> 64 fp32s (4x unrolling)

                // 4-way unrolled loop using 12 ZMM registers for better ILP
                for(; i < vec_size; i += 64) {
                    // Load 4 vectors of 16 FP16 values each (64 total)
                    __m256h in0 = _mm256_loadu_ph(&input[i]);
                    __m256h in1 = _mm256_loadu_ph(&input[i + 16]);
                    __m256h in2 = _mm256_loadu_ph(&input[i + 32]);
                    __m256h in3 = _mm256_loadu_ph(&input[i + 48]);

                    // Convert all 4 vectors to FP32 in parallel
                    __m512 out0 = _mm512_cvtph_ps(in0);
                    __m512 out1 = _mm512_cvtph_ps(in1);
                    __m512 out2 = _mm512_cvtph_ps(in2);
                    __m512 out3 = _mm512_cvtph_ps(in3);

                    // Store all 4 vectors
                    _mm512_storeu_ps(&output[i], out0);
                    _mm512_storeu_ps(&output[i + 16], out1);
                    _mm512_storeu_ps(&output[i + 32], out2);
                    _mm512_storeu_ps(&output[i + 48], out3);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (input.size() / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m256h in = _mm256_loadu_ph(&input[i]);
                    __m512 out = _mm512_cvtph_ps(in);
                    _mm512_storeu_ps(&output[i], out);
                }

                // Handle remaining elements
                for(; i < input.size(); i++) {
                    output[i] = fp16_to_fp32(input[i]);
                }

                return output;
            }

            // AVX512 optimized vector conversion FP32->FP16
            inline std::vector<uint16_t>
            convert_vector_f32_f16_avx512(const std::vector<float>& input) {
                std::vector<uint16_t> output;
                output.resize(input.size());

                size_t i = 0;
                size_t vec_size =
                        (input.size() / 64) * 64;  // Process 64 fp32s -> 64 fp16s (4x unrolling)

                // 4-way unrolled loop using 12 ZMM registers for better ILP
                for(; i < vec_size; i += 64) {
                    // Load 4 vectors of 16 FP32 values each (64 total)
                    __m512 in0 = _mm512_loadu_ps(&input[i]);
                    __m512 in1 = _mm512_loadu_ps(&input[i + 16]);
                    __m512 in2 = _mm512_loadu_ps(&input[i + 32]);
                    __m512 in3 = _mm512_loadu_ps(&input[i + 48]);

                    // Convert all 4 vectors to FP16 in parallel
                    __m256h out0 = _mm512_cvtps_ph(in0, _MM_FROUND_TO_NEAREST_INT);
                    __m256h out1 = _mm512_cvtps_ph(in1, _MM_FROUND_TO_NEAREST_INT);
                    __m256h out2 = _mm512_cvtps_ph(in2, _MM_FROUND_TO_NEAREST_INT);
                    __m256h out3 = _mm512_cvtps_ph(in3, _MM_FROUND_TO_NEAREST_INT);

                    // Store all 4 vectors
                    _mm256_storeu_ph(&output[i], out0);
                    _mm256_storeu_ph(&output[i + 16], out1);
                    _mm256_storeu_ph(&output[i + 32], out2);
                    _mm256_storeu_ph(&output[i + 48], out3);
                }

                // Handle remaining 16-element chunks
                size_t remaining_vec_size = (input.size() / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m512 in = _mm512_loadu_ps(&input[i]);
                    __m256h out = _mm512_cvtps_ph(in, _MM_FROUND_TO_NEAREST_INT);
                    _mm256_storeu_ph(&output[i], out);
                }

                // Handle remaining elements
                for(; i < input.size(); i++) {
                    output[i] = fp32_to_fp16(input[i]);
                }

                return output;
            }
#endif

            // High-level conversion functions that select the best implementation
            inline std::vector<float> convert_vector_f16_f32(const std::vector<uint16_t>& input) {
#if defined(USE_NEON)
                return convert_vector_f16_f32_neon(input);
#endif

#if defined(USE_AVX512)
                return convert_vector_f16_f32_avx512(input);
#endif

                // Fallback scalar implementation
                std::vector<float> output;
                output.resize(input.size());
                for(size_t i = 0; i < input.size(); i++) {
                    output[i] = fp16_to_fp32(input[i]);
                }
                return output;
            }

            inline std::vector<uint16_t> convert_vector_f32_f16(const std::vector<float>& input) {
#if defined(USE_NEON)
                return convert_vector_f32_f16_neon(input);
#endif

#if defined(USE_AVX512)
                return convert_vector_f32_f16_avx512(input);
#endif

                // Fallback scalar implementation
                std::vector<uint16_t> output;
                output.resize(input.size());
                for(size_t i = 0; i < input.size(); i++) {
                    output[i] = fp32_to_fp16(input[i]);
                }
                return output;
            }

            inline std::vector<uint16_t>
            convert_vector_f32_f16_scaled(const std::vector<float>& input, float scale) {
                if(scale == 1.0f) {
                    return convert_vector_f32_f16(input);
                }

                std::vector<uint16_t> output;
                output.resize(input.size());

                size_t i = 0;

#if defined(USE_NEON)
                const float32x4_t s = vdupq_n_f32(scale);
                size_t vec_size = (input.size() / 16) * 16;  // 16 floats per iteration (4x unroll)
                for(; i < vec_size; i += 16) {
                    float32x4_t in0 = vmulq_f32(vld1q_f32(&input[i]), s);
                    float32x4_t in1 = vmulq_f32(vld1q_f32(&input[i + 4]), s);
                    float32x4_t in2 = vmulq_f32(vld1q_f32(&input[i + 8]), s);
                    float32x4_t in3 = vmulq_f32(vld1q_f32(&input[i + 12]), s);

                    float16x4_t out0 = vcvt_f16_f32(in0);
                    float16x4_t out1 = vcvt_f16_f32(in1);
                    float16x4_t out2 = vcvt_f16_f32(in2);
                    float16x4_t out3 = vcvt_f16_f32(in3);

                    vst1_f16(reinterpret_cast<__fp16*>(&output[i]), out0);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 4]), out1);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 8]), out2);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i + 12]), out3);
                }

                size_t remaining_vec_size = (input.size() / 4) * 4;
                for(; i < remaining_vec_size; i += 4) {
                    float32x4_t in = vmulq_f32(vld1q_f32(&input[i]), s);
                    float16x4_t out = vcvt_f16_f32(in);
                    vst1_f16(reinterpret_cast<__fp16*>(&output[i]), out);
                }
#elif defined(USE_AVX512)
                const __m512 s = _mm512_set1_ps(scale);
                size_t vec_size = (input.size() / 64) * 64;  // 64 floats per iteration (4x unroll)
                for(; i < vec_size; i += 64) {
                    __m512 in0 = _mm512_mul_ps(_mm512_loadu_ps(&input[i]), s);
                    __m512 in1 = _mm512_mul_ps(_mm512_loadu_ps(&input[i + 16]), s);
                    __m512 in2 = _mm512_mul_ps(_mm512_loadu_ps(&input[i + 32]), s);
                    __m512 in3 = _mm512_mul_ps(_mm512_loadu_ps(&input[i + 48]), s);

                    __m256h out0 =
                            _mm512_cvtps_ph(in0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256h out1 =
                            _mm512_cvtps_ph(in1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256h out2 =
                            _mm512_cvtps_ph(in2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256h out3 =
                            _mm512_cvtps_ph(in3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                    _mm256_storeu_ph(&output[i], out0);
                    _mm256_storeu_ph(&output[i + 16], out1);
                    _mm256_storeu_ph(&output[i + 32], out2);
                    _mm256_storeu_ph(&output[i + 48], out3);
                }

                size_t remaining_vec_size = (input.size() / 16) * 16;
                for(; i < remaining_vec_size; i += 16) {
                    __m512 in = _mm512_mul_ps(_mm512_loadu_ps(&input[i]), s);
                    __m256h out =
                            _mm512_cvtps_ph(in, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm256_storeu_ph(&output[i], out);
                }
#endif

                for(; i < input.size(); i++) {
                    output[i] = fp32_to_fp16(input[i] * scale);
                }

                return output;
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                std::vector<uint16_t> fp16_vec = convert_vector_f32_f16(input);
                std::vector<uint8_t> result(fp16_vec.size() * sizeof(uint16_t));
                std::memcpy(result.data(), fp16_vec.data(), result.size());
                return result;
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                std::vector<uint16_t> fp16_data(dim);
                std::memcpy(fp16_data.data(), in, dim * sizeof(uint16_t));
                return convert_vector_f16_f32(fp16_data);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const uint16_t* pVect1 = (const uint16_t*)pVect1v;
                const uint16_t* pVect2 = (const uint16_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                size_t qty = params->dim;

                float res = 0;
                size_t i = 0;

#if defined(USE_NEON)
                float32x4_t sum = vdupq_n_f32(0.0f);

                // Process 16 elements per iteration (unrolled 2x8)
                for(; i + 16 <= qty; i += 16) {
                    float16x8_t v1_0 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x8_t v2_0 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i));
                    float16x8_t v1_1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i + 8));
                    float16x8_t v2_1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i + 8));

                    float16x8_t diff0 = vsubq_f16(v1_0, v2_0);
                    sum = vfmlalq_low_f16(sum, diff0, diff0);
                    sum = vfmlalq_high_f16(sum, diff0, diff0);

                    float16x8_t diff1 = vsubq_f16(v1_1, v2_1);
                    sum = vfmlalq_low_f16(sum, diff1, diff1);
                    sum = vfmlalq_high_f16(sum, diff1, diff1);
                }

                // Process remaining 8 elements
                for(; i + 8 <= qty; i += 8) {
                    float16x8_t v1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x8_t v2 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i));

                    float16x8_t diff = vsubq_f16(v1, v2);
                    sum = vfmlalq_low_f16(sum, diff, diff);
                    sum = vfmlalq_high_f16(sum, diff, diff);
                }

                // Process remaining 4 elements
                for(; i + 4 <= qty; i += 4) {
                    float16x4_t v1 = vld1_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x4_t v2 = vld1_f16(reinterpret_cast<const __fp16*>(pVect2 + i));
                    float16x4_t diff = vsub_f16(v1, v2);
                    float16x8_t diff_q = vcombine_f16(diff, vdup_n_f16(0));
                    sum = vfmlalq_low_f16(sum, diff_q, diff_q);
                }

                res = vaddvq_f32(sum);
#elif defined(USE_AVX512)
                __m512 sum = _mm512_setzero_ps();

                // Process 32 elements per iteration
                for(; i + 32 <= qty; i += 32) {
                    __m512h v1 = _mm512_loadu_ph(pVect1 + i);
                    __m512h v2 = _mm512_loadu_ph(pVect2 + i);

                    // Split into two 256-bit halves for conversion to float
                    __m256h v1_lo = _mm512_castph512_ph256(v1);
                    __m256h v2_lo = _mm512_castph512_ph256(v2);
                    __m256h v1_hi = _mm512_extracti32x8_epi32(_mm512_castph_si512(v1), 1);
                    __m256h v2_hi = _mm512_extracti32x8_epi32(_mm512_castph_si512(v2), 1);

                    __m512 f1_lo = _mm512_cvtph_ps(v1_lo);
                    __m512 f2_lo = _mm512_cvtph_ps(v2_lo);
                    __m512 diff_lo = _mm512_sub_ps(f1_lo, f2_lo);
                    sum = _mm512_fmadd_ps(diff_lo, diff_lo, sum);

                    __m512 f1_hi = _mm512_cvtph_ps(v1_hi);
                    __m512 f2_hi = _mm512_cvtph_ps(v2_hi);
                    __m512 diff_hi = _mm512_sub_ps(f1_hi, f2_hi);
                    sum = _mm512_fmadd_ps(diff_hi, diff_hi, sum);
                }

                res = _mm512_reduce_add_ps(sum);
#elif defined(USE_AVX2)
                __m256 sum = _mm256_setzero_ps();

                // Process 16 elements per iteration
                for(; i + 16 <= qty; i += 16) {
                    __m256i v1_raw =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pVect1 + i));
                    __m256i v2_raw =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pVect2 + i));

                    __m128i v1_lo = _mm256_castsi256_si128(v1_raw);
                    __m128i v2_lo = _mm256_castsi256_si128(v2_raw);
                    __m128i v1_hi = _mm256_extracti128_si256(v1_raw, 1);
                    __m128i v2_hi = _mm256_extracti128_si256(v2_raw, 1);

                    __m256 f1_lo = _mm256_cvtph_ps(v1_lo);
                    __m256 f2_lo = _mm256_cvtph_ps(v2_lo);
                    __m256 diff_lo = _mm256_sub_ps(f1_lo, f2_lo);
                    sum = _mm256_fmadd_ps(diff_lo, diff_lo, sum);

                    __m256 f1_hi = _mm256_cvtph_ps(v1_hi);
                    __m256 f2_hi = _mm256_cvtph_ps(v2_hi);
                    __m256 diff_hi = _mm256_sub_ps(f1_hi, f2_hi);
                    sum = _mm256_fmadd_ps(diff_hi, diff_hi, sum);
                }

                // Horizontal sum
                __m128 sum_lo = _mm256_castps256_ps128(sum);
                __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
                sum_lo = _mm_add_ps(sum_lo, sum_hi);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                res = _mm_cvtss_f32(sum_lo);
#elif defined(USE_SVE2)
                svfloat32_t sum0 = svdup_f32(0.0f);
                svfloat32_t sum1 = svdup_f32(0.0f);
                svfloat32_t sum2 = svdup_f32(0.0f);
                svfloat32_t sum3 = svdup_f32(0.0f);
                svfloat32_t sum4 = svdup_f32(0.0f);
                svfloat32_t sum5 = svdup_f32(0.0f);
                svfloat32_t sum6 = svdup_f32(0.0f);
                svfloat32_t sum7 = svdup_f32(0.0f);

                uint64_t num_elements = svcnth();
                size_t unroll_stride = num_elements * 4;
                svbool_t pg_all = svptrue_b16();

                // Main unrolled loop
                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    // First vector
                    svfloat16_t v1_0 = svld1_f16(pg_all, (const __fp16*)(pVect1 + i));
                    svfloat16_t v2_0 = svld1_f16(pg_all, (const __fp16*)(pVect2 + i));
                    svfloat16_t diff_0 = svsub_f16_x(pg_all, v1_0, v2_0);

                    sum0 = svmlalb_f32(sum0, diff_0, diff_0);
                    sum1 = svmlalt_f32(sum1, diff_0, diff_0);

                    // Second vector
                    svfloat16_t v1_1 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + num_elements));
                    svfloat16_t v2_1 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + num_elements));
                    svfloat16_t diff_1 = svsub_f16_x(pg_all, v1_1, v2_1);

                    sum2 = svmlalb_f32(sum2, diff_1, diff_1);
                    sum3 = svmlalt_f32(sum3, diff_1, diff_1);

                    // Third vector
                    svfloat16_t v1_2 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + 2 * num_elements));
                    svfloat16_t v2_2 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + 2 * num_elements));
                    svfloat16_t diff_2 = svsub_f16_x(pg_all, v1_2, v2_2);

                    sum4 = svmlalb_f32(sum4, diff_2, diff_2);
                    sum5 = svmlalt_f32(sum5, diff_2, diff_2);

                    // Fourth vector
                    svfloat16_t v1_3 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + 3 * num_elements));
                    svfloat16_t v2_3 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + 3 * num_elements));
                    svfloat16_t diff_3 = svsub_f16_x(pg_all, v1_3, v2_3);

                    sum6 = svmlalb_f32(sum6, diff_3, diff_3);
                    sum7 = svmlalt_f32(sum7, diff_3, diff_3);
                }

                svfloat32_t s0 = svadd_f32_x(svptrue_b32(), sum0, sum1);
                svfloat32_t s1 = svadd_f32_x(svptrue_b32(), sum2, sum3);
                svfloat32_t s2 = svadd_f32_x(svptrue_b32(), sum4, sum5);
                svfloat32_t s3 = svadd_f32_x(svptrue_b32(), sum6, sum7);
                s0 = svadd_f32_x(svptrue_b32(), s0, s1);
                s2 = svadd_f32_x(svptrue_b32(), s2, s3);
                svfloat32_t sum = svadd_f32_x(svptrue_b32(), s0, s2);

                svbool_t pg = svwhilelt_b16(i, qty);
                while(svptest_any(svptrue_b16(), pg)) {
                    svfloat16_t v1 = svld1_f16(pg, (const __fp16*)(pVect1 + i));
                    svfloat16_t v2 = svld1_f16(pg, (const __fp16*)(pVect2 + i));
                    svfloat16_t diff = svsub_f16_z(pg, v1, v2);

                    sum = svmlalb_f32(sum, diff, diff);
                    sum = svmlalt_f32(sum, diff, diff);

                    i += svcnth();
                    pg = svwhilelt_b16(i, qty);
                }
                res = svaddv_f32(svptrue_b32(), sum);
#endif

                for(; i < qty; i++) {
                    float v1 = fp16_to_fp32(pVect1[i]);
                    float v2 = fp16_to_fp32(pVect2[i]);
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
                const uint16_t* pVect1 = (const uint16_t*)pVect1v;
                const uint16_t* pVect2 = (const uint16_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                size_t qty = params->dim;

                float res = 0;
                size_t i = 0;

#if defined(USE_NEON)
                // Dual accumulators: improves ILP/throughput on Apple Silicon.
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);

                // Process 16 elements per iteration
                for(; i + 16 <= qty; i += 16) {
                    float16x8_t v1_0 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x8_t v2_0 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i));
                    float16x8_t v1_1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i + 8));
                    float16x8_t v2_1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i + 8));

                    sum0 = vfmlalq_low_f16(sum0, v1_0, v2_0);
                    sum0 = vfmlalq_high_f16(sum0, v1_0, v2_0);
                    sum1 = vfmlalq_low_f16(sum1, v1_1, v2_1);
                    sum1 = vfmlalq_high_f16(sum1, v1_1, v2_1);
                }

                // Process remaining 8 elements
                for(; i + 8 <= qty; i += 8) {
                    float16x8_t v1 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x8_t v2 = vld1q_f16(reinterpret_cast<const __fp16*>(pVect2 + i));

                    sum0 = vfmlalq_low_f16(sum0, v1, v2);
                    sum0 = vfmlalq_high_f16(sum0, v1, v2);
                }

                // Process remaining 4 elements
                for(; i + 4 <= qty; i += 4) {
                    float16x4_t v1 = vld1_f16(reinterpret_cast<const __fp16*>(pVect1 + i));
                    float16x4_t v2 = vld1_f16(reinterpret_cast<const __fp16*>(pVect2 + i));
                    float16x8_t v1_q = vcombine_f16(v1, vdup_n_f16(0));
                    float16x8_t v2_q = vcombine_f16(v2, vdup_n_f16(0));
                    sum0 = vfmlalq_low_f16(sum0, v1_q, v2_q);
                }

                res = vaddvq_f32(vaddq_f32(sum0, sum1));
#elif defined(USE_AVX512)
                __m512 sum = _mm512_setzero_ps();

                // Process 32 elements per iteration
                for(; i + 32 <= qty; i += 32) {
                    __m512h v1 = _mm512_loadu_ph(pVect1 + i);
                    __m512h v2 = _mm512_loadu_ph(pVect2 + i);

                    __m256h v1_lo = _mm512_castph512_ph256(v1);
                    __m256h v2_lo = _mm512_castph512_ph256(v2);
                    __m256h v1_hi = _mm512_extracti32x8_epi32(_mm512_castph_si512(v1), 1);
                    __m256h v2_hi = _mm512_extracti32x8_epi32(_mm512_castph_si512(v2), 1);

                    __m512 f1_lo = _mm512_cvtph_ps(v1_lo);
                    __m512 f2_lo = _mm512_cvtph_ps(v2_lo);
                    sum = _mm512_fmadd_ps(f1_lo, f2_lo, sum);

                    __m512 f1_hi = _mm512_cvtph_ps(v1_hi);
                    __m512 f2_hi = _mm512_cvtph_ps(v2_hi);
                    sum = _mm512_fmadd_ps(f1_hi, f2_hi, sum);
                }

                res = _mm512_reduce_add_ps(sum);
#elif defined(USE_AVX2)
                __m256 sum = _mm256_setzero_ps();

                // Process 16 elements per iteration
                for(; i + 16 <= qty; i += 16) {
                    __m256i v1_raw =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pVect1 + i));
                    __m256i v2_raw =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pVect2 + i));

                    __m128i v1_lo = _mm256_castsi256_si128(v1_raw);
                    __m128i v2_lo = _mm256_castsi256_si128(v2_raw);
                    __m128i v1_hi = _mm256_extracti128_si256(v1_raw, 1);
                    __m128i v2_hi = _mm256_extracti128_si256(v2_raw, 1);

                    __m256 f1_lo = _mm256_cvtph_ps(v1_lo);
                    __m256 f2_lo = _mm256_cvtph_ps(v2_lo);
                    sum = _mm256_fmadd_ps(f1_lo, f2_lo, sum);

                    __m256 f1_hi = _mm256_cvtph_ps(v1_hi);
                    __m256 f2_hi = _mm256_cvtph_ps(v2_hi);
                    sum = _mm256_fmadd_ps(f1_hi, f2_hi, sum);
                }

                // Horizontal sum
                __m128 sum_lo = _mm256_castps256_ps128(sum);
                __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
                sum_lo = _mm_add_ps(sum_lo, sum_hi);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                res = _mm_cvtss_f32(sum_lo);
#elif defined(USE_SVE2)
                svfloat32_t sum0 = svdup_f32(0.0f);
                svfloat32_t sum1 = svdup_f32(0.0f);
                svfloat32_t sum2 = svdup_f32(0.0f);
                svfloat32_t sum3 = svdup_f32(0.0f);
                svfloat32_t sum4 = svdup_f32(0.0f);
                svfloat32_t sum5 = svdup_f32(0.0f);
                svfloat32_t sum6 = svdup_f32(0.0f);
                svfloat32_t sum7 = svdup_f32(0.0f);

                uint64_t num_elements = svcnth();
                size_t unroll_stride = num_elements * 4;
                svbool_t pg_all = svptrue_b16();

                // Main unrolled loop
                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    // First vector
                    svfloat16_t v1_0 = svld1_f16(pg_all, (const __fp16*)(pVect1 + i));
                    svfloat16_t v2_0 = svld1_f16(pg_all, (const __fp16*)(pVect2 + i));
                    sum0 = svmlalb_f32(sum0, v1_0, v2_0);
                    sum1 = svmlalt_f32(sum1, v1_0, v2_0);

                    // Second vector
                    svfloat16_t v1_1 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + num_elements));
                    svfloat16_t v2_1 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + num_elements));
                    sum2 = svmlalb_f32(sum2, v1_1, v2_1);
                    sum3 = svmlalt_f32(sum3, v1_1, v2_1);

                    // Third vector
                    svfloat16_t v1_2 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + 2 * num_elements));
                    svfloat16_t v2_2 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + 2 * num_elements));
                    sum4 = svmlalb_f32(sum4, v1_2, v2_2);
                    sum5 = svmlalt_f32(sum5, v1_2, v2_2);

                    // Fourth vector
                    svfloat16_t v1_3 =
                            svld1_f16(pg_all, (const __fp16*)(pVect1 + i + 3 * num_elements));
                    svfloat16_t v2_3 =
                            svld1_f16(pg_all, (const __fp16*)(pVect2 + i + 3 * num_elements));
                    sum6 = svmlalb_f32(sum6, v1_3, v2_3);
                    sum7 = svmlalt_f32(sum7, v1_3, v2_3);
                }

                svfloat32_t s0 = svadd_f32_x(svptrue_b32(), sum0, sum1);
                svfloat32_t s1 = svadd_f32_x(svptrue_b32(), sum2, sum3);
                svfloat32_t s2 = svadd_f32_x(svptrue_b32(), sum4, sum5);
                svfloat32_t s3 = svadd_f32_x(svptrue_b32(), sum6, sum7);
                s0 = svadd_f32_x(svptrue_b32(), s0, s1);
                s2 = svadd_f32_x(svptrue_b32(), s2, s3);
                svfloat32_t sum = svadd_f32_x(svptrue_b32(), s0, s2);

                svbool_t pg = svwhilelt_b16(i, qty);
                while(svptest_any(svptrue_b16(), pg)) {
                    svfloat16_t v1 = svld1_f16(pg, (const __fp16*)(pVect1 + i));
                    svfloat16_t v2 = svld1_f16(pg, (const __fp16*)(pVect2 + i));

                    sum = svmlalb_f32(sum, v1, v2);
                    sum = svmlalt_f32(sum, v1, v2);

                    i += svcnth();
                    pg = svwhilelt_b16(i, qty);
                }
                res = svaddv_f32(svptrue_b32(), sum);
#endif

                for(; i < qty; i++) {
                    float v1 = fp16_to_fp32(pVect1[i]);
                    float v2 = fp16_to_fp32(pVect2[i]);
                    res += v1 * v2;
                }
                return res;
            }

            static float
            InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float CosineSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                // Vectors are guaranteed normalized => cosine similarity == inner product.
                // This reuses the same SIMD paths (NEON/AVX512/AVX2) as InnerProductSim.
                return InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float Cosine(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - CosineSim(pVect1v, pVect2v, qty_ptr);
            }

            static std::vector<uint8_t> quantize_to_int8(const void* in, size_t dim) {
                const uint16_t* input = static_cast<const uint16_t*>(in);
                // Calculate storage size: dim bytes for data + 4 bytes for scale
                size_t buffer_size = dim * sizeof(int8_t) + sizeof(float);
                std::vector<uint8_t> buffer(buffer_size);
                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                float scale;
                float abs_max = 0.0f;

#if defined(USE_AVX512)
                size_t i = 0;
                __m512 max_vec = _mm512_setzero_ps();
                static const __m512 sign_mask_vec = _mm512_set1_ps(-0.0f);

                for(; i + 32 <= dim; i += 32) {
                    __m256i in_half1 = _mm256_loadu_si256((const __m256i*)&input[i]);
                    __m256i in_half2 = _mm256_loadu_si256((const __m256i*)&input[i + 16]);

                    __m512 v1 = _mm512_cvtph_ps(in_half1);
                    __m512 v2 = _mm512_cvtph_ps(in_half2);

                    v1 = _mm512_andnot_ps(sign_mask_vec, v1);
                    v2 = _mm512_andnot_ps(sign_mask_vec, v2);

                    max_vec = _mm512_max_ps(max_vec, v1);
                    max_vec = _mm512_max_ps(max_vec, v2);
                }
                abs_max = _mm512_reduce_max_ps(max_vec);

                for(; i < dim; ++i) {
                    float val = std::abs(fp16_to_fp32(input[i]));
                    if(val > abs_max) {
                        abs_max = val;
                    }
                }
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                scale = 127.0f / abs_max;
                __m512 scale_vec = _mm512_set1_ps(scale);

                i = 0;
                for(; i + 64 <= dim; i += 64) {
                    __m256i h0 = _mm256_loadu_si256((const __m256i*)&input[i]);
                    __m256i h1 = _mm256_loadu_si256((const __m256i*)&input[i + 16]);
                    __m256i h2 = _mm256_loadu_si256((const __m256i*)&input[i + 32]);
                    __m256i h3 = _mm256_loadu_si256((const __m256i*)&input[i + 48]);

                    __m512 f0 = _mm512_cvtph_ps(h0);
                    __m512 f1 = _mm512_cvtph_ps(h1);
                    __m512 f2 = _mm512_cvtph_ps(h2);
                    __m512 f3 = _mm512_cvtph_ps(h3);

                    f0 = _mm512_mul_ps(f0, scale_vec);
                    f1 = _mm512_mul_ps(f1, scale_vec);
                    f2 = _mm512_mul_ps(f2, scale_vec);
                    f3 = _mm512_mul_ps(f3, scale_vec);

                    __m512i i0 = _mm512_cvtps_epi32(f0);
                    __m512i i1 = _mm512_cvtps_epi32(f1);
                    __m512i i2 = _mm512_cvtps_epi32(f2);
                    __m512i i3 = _mm512_cvtps_epi32(f3);

                    __m128i p0 = _mm512_cvtepi32_epi8(i0);
                    __m128i p1 = _mm512_cvtepi32_epi8(i1);
                    __m128i p2 = _mm512_cvtepi32_epi8(i2);
                    __m128i p3 = _mm512_cvtepi32_epi8(i3);

                    _mm_storeu_si128((__m128i*)&data_ptr[i], p0);
                    _mm_storeu_si128((__m128i*)&data_ptr[i+16], p1);
                    _mm_storeu_si128((__m128i*)&data_ptr[i+32], p2);
                    _mm_storeu_si128((__m128i*)&data_ptr[i+48], p3);
                }
#elif defined(USE_AVX2)
                size_t i = 0;
                __m256 max_vec = _mm256_setzero_ps();
                static const __m256 sign_mask_vec = _mm256_set1_ps(-0.0f);

                for(; i + 16 <= dim; i += 16) {
                    __m128i h1 = _mm_loadu_si128((const __m128i*)&input[i]);
                    __m128i h2 = _mm_loadu_si128((const __m128i*)&input[i + 8]);

                    __m256 v1 = _mm256_cvtph_ps(h1);
                    __m256 v2 = _mm256_cvtph_ps(h2);

                    v1 = _mm256_andnot_ps(sign_mask_vec, v1);
                    v2 = _mm256_andnot_ps(sign_mask_vec, v2);

                    max_vec = _mm256_max_ps(max_vec, v1);
                    max_vec = _mm256_max_ps(max_vec, v2);
                }

                __m128 max128 = _mm_max_ps(_mm256_castsi256_si128(_mm256_castps_si256(max_vec)),
                                           _mm256_extractf128_ps(max_vec, 1));
                max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
                max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0x1));
                abs_max = _mm_cvtss_f32(max128);

                for(; i < dim; ++i) {
                    float val = std::abs(fp16_to_fp32(input[i]));
                    if(val > abs_max) {
                        abs_max = val;
                    }
                }
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                scale = 127.0f / abs_max;
                __m256 scale_vec = _mm256_set1_ps(scale);

                i = 0;
                for(; i + 32 <= dim; i += 32) {
                    __m128i h0 = _mm_loadu_si128((const __m128i*)&input[i]);
                    __m128i h1 = _mm_loadu_si128((const __m128i*)&input[i + 8]);
                    __m128i h2 = _mm_loadu_si128((const __m128i*)&input[i + 16]);
                    __m128i h3 = _mm_loadu_si128((const __m128i*)&input[i + 24]);

                    __m256 f0 = _mm256_cvtph_ps(h0);
                    __m256 f1 = _mm256_cvtph_ps(h1);
                    __m256 f2 = _mm256_cvtph_ps(h2);
                    __m256 f3 = _mm256_cvtph_ps(h3);

                    f0 = _mm256_mul_ps(f0, scale_vec);
                    f1 = _mm256_mul_ps(f1, scale_vec);
                    f2 = _mm256_mul_ps(f2, scale_vec);
                    f3 = _mm256_mul_ps(f3, scale_vec);

                    __m256i i0 = _mm256_cvtps_epi32(f0);
                    __m256i i1 = _mm256_cvtps_epi32(f1);
                    __m256i i2 = _mm256_cvtps_epi32(f2);
                    __m256i i3 = _mm256_cvtps_epi32(f3);

                    __m256i p01 = _mm256_packs_epi32(i0, i1);
                    __m256i p23 = _mm256_packs_epi32(i2, i3);

                    p01 = _mm256_permute4x64_epi64(p01, _MM_SHUFFLE(3, 1, 2, 0));
                    p23 = _mm256_permute4x64_epi64(p23, _MM_SHUFFLE(3, 1, 2, 0));

                    __m256i p = _mm256_packs_epi16(p01, p23);
                    p = _mm256_permute4x64_epi64(p, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm256_storeu_si256((__m256i*)&data_ptr[i], p);
                }
#elif defined(USE_SVE2)
                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dim);

                svfloat32_t max_val = svdup_f32(0.0f);
                while(svptest_any(svptrue_b32(), pg)) {
                    svbool_t pg_load = svwhilelt_b16(i, dim);
                    svfloat16_t in_h =
                            svld1_f16(pg_load, reinterpret_cast<const __fp16*>(&input[i]));
                    svfloat32_t val = svcvt_f32_f16_x(pg, in_h);
                    val = svabs_f32_x(pg, val);
                    max_val = svmax_f32_m(pg, max_val, val);
                    i += svcntw();
                    pg = svwhilelt_b32(i, dim);
                }
                abs_max = svmaxv_f32(svptrue_b32(), max_val);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                scale = 127.0f / abs_max;

                i = 0;
                pg = svwhilelt_b32(i, dim);
                svfloat32_t scale_vec = svdup_f32(scale);
                while(svptest_any(svptrue_b32(), pg)) {
                    svbool_t pg_load = svwhilelt_b16(i, dim);
                    svfloat16_t in_h =
                            svld1_f16(pg_load, reinterpret_cast<const __fp16*>(&input[i]));
                    svfloat32_t val = svcvt_f32_f16_x(pg, in_h);
                    val = svmul_f32_x(pg, val, scale_vec);

                    svint32_t i_val = svcvt_s32_f32_x(pg, val);
                    i_val = svmin_s32_x(pg, i_val, svdup_s32(127));
                    i_val = svmax_s32_x(pg, i_val, svdup_s32(-127));

                    svst1b_s32(pg, &data_ptr[i], i_val);
                    i += svcntw();
                    pg = svwhilelt_b32(i, dim);
                }
#elif defined(USE_NEON)
                float32x4_t max_vec = vdupq_n_f32(0.0f);
                size_t i = 0;
                // Pass 1: Find absolute max
                for(; i + 16 <= dim; i += 16) {
                    float16x4_t in0 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i]));
                    float16x4_t in1 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 4]));
                    float16x4_t in2 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 8]));
                    float16x4_t in3 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 12]));

                    float32x4_t v0 = vabsq_f32(vcvt_f32_f16(in0));
                    float32x4_t v1 = vabsq_f32(vcvt_f32_f16(in1));
                    float32x4_t v2 = vabsq_f32(vcvt_f32_f16(in2));
                    float32x4_t v3 = vabsq_f32(vcvt_f32_f16(in3));

                    max_vec = vmaxq_f32(max_vec, v0);
                    max_vec = vmaxq_f32(max_vec, v1);
                    max_vec = vmaxq_f32(max_vec, v2);
                    max_vec = vmaxq_f32(max_vec, v3);
                }
                abs_max = vmaxvq_f32(max_vec);

                // Handle remaining for max
                for(; i < dim; ++i) {
                    float val = std::abs(fp16_to_fp32(input[i]));
                    if(val > abs_max) {
                        abs_max = val;
                    }
                }

                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                scale = 127.0f / abs_max;
                float32x4_t scale_vec = vdupq_n_f32(scale);

                // Pass 2: Quantize
                i = 0;
                for(; i + 16 <= dim; i += 16) {
                    float16x4_t in0 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i]));
                    float16x4_t in1 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 4]));
                    float16x4_t in2 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 8]));
                    float16x4_t in3 = vld1_f16(reinterpret_cast<const __fp16*>(&input[i + 12]));

                    // Convert to f32 and scale
                    float32x4_t f0 = vmulq_f32(vcvt_f32_f16(in0), scale_vec);
                    float32x4_t f1 = vmulq_f32(vcvt_f32_f16(in1), scale_vec);
                    float32x4_t f2 = vmulq_f32(vcvt_f32_f16(in2), scale_vec);
                    float32x4_t f3 = vmulq_f32(vcvt_f32_f16(in3), scale_vec);

                    // Convert to int32 with rounding
                    int32x4_t i0 = vcvtaq_s32_f32(f0);
                    int32x4_t i1 = vcvtaq_s32_f32(f1);
                    int32x4_t i2 = vcvtaq_s32_f32(f2);
                    int32x4_t i3 = vcvtaq_s32_f32(f3);

                    // Pack 32->16
                    int16x8_t p0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                    int16x8_t p1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));

                    // Pack 16->8
                    int8x16_t out = vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1));
                    vst1q_s8(&data_ptr[i], out);
                }
#else
                size_t i = 0;
#endif

                // Scalar fallback loop (shared)
                for(; i < dim; ++i) {
                    float val = std::abs(fp16_to_fp32(input[i]));
                    if(val > abs_max) {
                        abs_max = val;
                    }
                }

                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                scale = 127.0f / abs_max;

                for(; i < dim; ++i) {
                    float val = fp16_to_fp32(input[i]);
                    data_ptr[i] = static_cast<int8_t>(std::round(val * scale));
                }

                float* scale_ptr = reinterpret_cast<float*>(buffer.data() + dim);
                *scale_ptr = abs_max / 127.0f;

                return buffer;
            }

        }  // namespace float16

        class Float16Quantizer : public Quantizer {
        public:
            std::string name() const override { return "float16"; }
            QuantizationLevel level() const override { return QuantizationLevel::FP16; }

            QuantizerDispatch getDispatch() const override {
                QuantizerDispatch d;
                d.dist_l2 = &float16::L2Sqr;
                d.dist_ip = &float16::InnerProduct;
                d.dist_cosine = &float16::Cosine;
                d.sim_l2 = &float16::L2SqrSim;
                d.sim_ip = &float16::InnerProductSim;
                d.sim_cosine = &float16::CosineSim;
                d.quantize = &float16::quantize;
                d.dequantize = &float16::dequantize;
                d.quantize_to_int8 = &float16::quantize_to_int8;
                d.get_storage_size = &float16::get_storage_size;
                d.extract_inv_scale = &float16::extract_inv_scale;
                return d;
            }
        };

        // Register FP16
        static RegisterQuantizer
                reg_fp16(QuantizationLevel::FP16, "float16", std::make_shared<Float16Quantizer>());

    }  // namespace quant
}  // namespace ndd
