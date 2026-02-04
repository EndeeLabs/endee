#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "common.hpp"

#if defined(USE_NEON)
#    include <arm_neon.h>
#endif

#if defined(USE_AVX512) || defined(USE_AVX2)
#    include <immintrin.h>
#endif

#if defined(USE_SVE2)
#    include <arm_sve.h>
#endif

namespace ndd {
    namespace quant {
        namespace binary {

            // Calculate storage size in bytes (padded to multiple of 64 bits / 8 bytes)
            inline size_t get_storage_size(size_t dimension) {
                size_t num_uint64 = (dimension + 63) / 64;
                return num_uint64 * 8;
            }

            // No scale for binary quantization
            inline float extract_inv_scale(const uint8_t* buffer, size_t dimension) {
                return 1.0f;
            }

#if defined(USE_AVX512)
            inline std::vector<uint8_t> quantize_avx512(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size, 0);

                size_t i = 0;
                __m512 zero = _mm512_setzero_ps();

                // Process 16 floats -> 16 bits (2 bytes)
                for(; i + 16 <= dimension; i += 16) {
                    __m512 v = _mm512_loadu_ps(&input[i]);
                    __mmask16 m = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);

                    // Store 16 bits
                    uint16_t bits = (uint16_t)m;
                    *reinterpret_cast<uint16_t*>(buffer.data() + i / 8) = bits;
                }

                // Scalar fallback
                uint64_t* data_ptr = reinterpret_cast<uint64_t*>(buffer.data());
                for(; i < dimension; ++i) {
                    if(input[i] > 0) {
                        size_t chunk_idx = i / 64;
                        size_t bit_idx = i % 64;
                        data_ptr[chunk_idx] |= (1ULL << bit_idx);
                    }
                }

                return buffer;
            }
#endif

#if defined(USE_NEON)
            inline std::vector<uint8_t> quantize_neon(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                const size_t dimension = input.size();
                const size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size, 0);

                uint64_t* out64 = reinterpret_cast<uint64_t*>(buffer.data());
                const float32x4_t v_zero = vdupq_n_f32(0.0f);

                // Powers of 2 for bit packing within each byte
                const uint8_t powers_data[] = {1, 2, 4, 8, 16, 32, 64, 128};
                const uint8x8_t v_powers = vld1_u8(powers_data);

                size_t i = 0;

                // Process 64 floats -> 8 bytes (one uint64_t) per iteration
                // This matches the granularity of Hamming distance (8 uint64 per NEON iter)
                for(; i + 64 <= dimension; i += 64) {
                    uint64_t result = 0;

// Unroll 8x: each iteration produces 1 byte
#    pragma unroll
                    for(int j = 0; j < 8; ++j) {
                        const float* ptr = &input[i + j * 8];
                        float32x4_t v0 = vld1q_f32(ptr);
                        float32x4_t v1 = vld1q_f32(ptr + 4);

                        // Compare > 0
                        uint32x4_t m0 = vcgtq_f32(v0, v_zero);
                        uint32x4_t m1 = vcgtq_f32(v1, v_zero);

                        // Narrow 32 -> 16 -> 8 bit masks
                        uint16x4_t n0 = vmovn_u32(m0);
                        uint16x4_t n1 = vmovn_u32(m1);
                        uint8x8_t n_8 = vmovn_u16(vcombine_u16(n0, n1));

                        // AND with powers and horizontal sum to get byte
                        uint8_t byte = vaddv_u8(vand_u8(n_8, v_powers));
                        result |= ((uint64_t)byte << (j * 8));
                    }

                    out64[i / 64] = result;
                }

                // Process remaining 8 floats -> 1 byte at a time
                for(; i + 8 <= dimension; i += 8) {
                    float32x4_t v0 = vld1q_f32(&input[i]);
                    float32x4_t v1 = vld1q_f32(&input[i + 4]);

                    uint32x4_t m0 = vcgtq_f32(v0, v_zero);
                    uint32x4_t m1 = vcgtq_f32(v1, v_zero);

                    uint16x4_t n0 = vmovn_u32(m0);
                    uint16x4_t n1 = vmovn_u32(m1);
                    uint8x8_t n_8 = vmovn_u16(vcombine_u16(n0, n1));

                    buffer[i / 8] = vaddv_u8(vand_u8(n_8, v_powers));
                }

                // Scalar fallback for remaining elements
                for(; i < dimension; ++i) {
                    if(input[i] > 0) {
                        const size_t chunk_idx = i / 64;
                        const size_t bit_idx = i % 64;
                        out64[chunk_idx] |= (1ULL << bit_idx);
                    }
                }

                return buffer;
            }
#endif

#if defined(USE_AVX2)
            inline std::vector<uint8_t> quantize_avx2(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size, 0);

                size_t i = 0;
                __m256 zero = _mm256_setzero_ps();

                // Process 8 floats -> 8 bits (1 byte)
                for(; i + 8 <= dimension; i += 8) {
                    __m256 v = _mm256_loadu_ps(&input[i]);
                    __m256 cmp = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
                    int mask = _mm256_movemask_ps(cmp);
                    buffer[i / 8] = (uint8_t)mask;
                }

                // Scalar fallback
                uint64_t* data_ptr = reinterpret_cast<uint64_t*>(buffer.data());
                for(; i < dimension; ++i) {
                    if(input[i] > 0) {
                        size_t chunk_idx = i / 64;
                        size_t bit_idx = i % 64;
                        data_ptr[chunk_idx] |= (1ULL << bit_idx);
                    }
                }

                return buffer;
            }
#endif

            // Quantize FP32 vector to Binary (packed bits)
            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
#if defined(USE_AVX512)
                return quantize_avx512(input);
#elif defined(USE_AVX2)
                return quantize_avx2(input);
#elif defined(USE_SVE2) && defined(USE_NEON)
                return quantize_neon(input);
#elif defined(USE_NEON)
                return quantize_neon(input);
#else
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                size_t dimension = input.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size, 0);

                uint64_t* data_ptr = reinterpret_cast<uint64_t*>(buffer.data());

                for(size_t i = 0; i < dimension; ++i) {
                    if(input[i] > 0) {
                        size_t chunk_idx = i / 64;
                        size_t bit_idx = i % 64;
                        data_ptr[chunk_idx] |= (1ULL << bit_idx);
                    }
                }

                return buffer;
#endif
            }

#if defined(USE_AVX512)
            inline std::vector<float> dequantize_avx512(const uint8_t* buffer, size_t dimension) {
                std::vector<float> output(dimension);

                size_t i = 0;
                __m512 ones = _mm512_set1_ps(1.0f);
                __m512 neg_ones = _mm512_set1_ps(-1.0f);

                for(; i + 16 <= dimension; i += 16) {
                    uint16_t bits = *reinterpret_cast<const uint16_t*>(buffer + i / 8);
                    __mmask16 m = (__mmask16)bits;

                    __m512 v = _mm512_mask_blend_ps(m, neg_ones, ones);
                    _mm512_storeu_ps(&output[i], v);
                }

                // Scalar fallback
                const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(buffer);
                for(; i < dimension; ++i) {
                    size_t chunk_idx = i / 64;
                    size_t bit_idx = i % 64;
                    if(data_ptr[chunk_idx] & (1ULL << bit_idx)) {
                        output[i] = 1.0f;
                    } else {
                        output[i] = -1.0f;
                    }
                }
                return output;
            }
#endif

#if defined(USE_AVX2)
            inline std::vector<float> dequantize_avx2(const uint8_t* buffer, size_t dimension) {
                std::vector<float> output(dimension);

                size_t i = 0;
                __m256 ones = _mm256_set1_ps(1.0f);
                __m256 neg_ones = _mm256_set1_ps(-1.0f);
                __m256i bit_masks = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

                for(; i + 8 <= dimension; i += 8) {
                    uint8_t byte = buffer[i / 8];
                    __m256i v_byte = _mm256_set1_epi32(byte);

                    __m256i masked = _mm256_and_si256(v_byte, bit_masks);
                    __m256i cmp = _mm256_cmpeq_epi32(
                            masked, _mm256_setzero_si256());  // -1 if 0 (bit not set), 0 if set

                    // cmp is -1 (all ones) if bit is 0 -> we want -1.0f.
                    // cmp is 0 if bit is 1 -> we want 1.0f.
                    // blendv_ps(a, b, mask) selects b if mask has MSB set.
                    // So if cmp is -1 (MSB set), we select b.
                    // We want -1.0f if cmp is -1.
                    // So blendv_ps(ones, neg_ones, cmp).

                    __m256 res = _mm256_blendv_ps(ones, neg_ones, _mm256_castsi256_ps(cmp));
                    _mm256_storeu_ps(&output[i], res);
                }

                // Scalar fallback
                const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(buffer);
                for(; i < dimension; ++i) {
                    size_t chunk_idx = i / 64;
                    size_t bit_idx = i % 64;
                    if(data_ptr[chunk_idx] & (1ULL << bit_idx)) {
                        output[i] = 1.0f;
                    } else {
                        output[i] = -1.0f;
                    }
                }
                return output;
            }
#endif

#if defined(USE_NEON)
            inline std::vector<float> dequantize_neon(const uint8_t* buffer, size_t dimension) {
                std::vector<float> output(dimension);
                size_t i = 0;
                float32x4_t v_one = vdupq_n_f32(1.0f);
                float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
                const uint8_t powers_data[] = {1, 2, 4, 8, 16, 32, 64, 128};
                uint8x8_t v_powers = vld1_u8(powers_data);

                for(; i + 8 <= dimension; i += 8) {
                    uint8_t byte = buffer[i / 8];
                    uint8x8_t v_byte = vdup_n_u8(byte);

                    // Test bits: returns 0xFF if bit is set, 0x00 otherwise
                    uint8x8_t v_mask8 = vtst_u8(v_byte, v_powers);

                    // Expand 8-bit masks to 16-bit (signed to preserve 0xFF -> 0xFFFF)
                    int16x8_t v_mask16 = vmovl_s8(vreinterpret_s8_u8(v_mask8));

                    // Expand 16-bit masks to 32-bit (signed to preserve 0xFFFF -> 0xFFFFFFFF)
                    int32x4_t v_mask32_0 = vmovl_s16(vget_low_s16(v_mask16));
                    int32x4_t v_mask32_1 = vmovl_s16(vget_high_s16(v_mask16));

                    // Select 1.0f or -1.0f based on mask
                    float32x4_t res0 =
                            vbslq_f32(vreinterpretq_u32_s32(v_mask32_0), v_one, v_neg_one);
                    float32x4_t res1 =
                            vbslq_f32(vreinterpretq_u32_s32(v_mask32_1), v_one, v_neg_one);

                    vst1q_f32(&output[i], res0);
                    vst1q_f32(&output[i + 4], res1);
                }

                // Scalar fallback
                const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(buffer);
                for(; i < dimension; ++i) {
                    size_t chunk_idx = i / 64;
                    size_t bit_idx = i % 64;
                    if(data_ptr[chunk_idx] & (1ULL << bit_idx)) {
                        output[i] = 1.0f;
                    } else {
                        output[i] = -1.0f;
                    }
                }
                return output;
            }
#endif

#if defined(USE_SVE2)
            inline std::vector<float> dequantize_sve(const uint8_t* buffer, size_t dimension) {
                std::vector<float> output(dimension);

                size_t i = 0;
                svbool_t pg = svwhilelt_b32(i, dimension);
                svfloat32_t ones = svdup_f32(1.0f);
                svfloat32_t neg_ones = svdup_f32(-1.0f);

                while(svptest_any(svptrue_b32(), pg)) {
                    svuint32_t idx = svindex_u32(i, 1);
                    svuint32_t byte_idx = svlsr_n_u32_x(pg, idx, 3);
                    svuint32_t bit_idx = svand_n_u32_x(pg, idx, 7);

                    // Exact match for the signature found in the header:
                    // svuint32_t svld1ub_gather_u32offset_u32(svbool_t, uint8_t const *,
                    // svuint32_t);
                    svuint32_t val = svld1ub_gather_u32offset_u32(pg, buffer, byte_idx);

                    svuint32_t mask = svlsl_u32_x(pg, svdup_u32(1), bit_idx);
                    svuint32_t masked = svand_u32_x(pg, val, mask);
                    svbool_t is_set = svcmpne_u32(pg, masked, svdup_u32(0));

                    svfloat32_t res = svsel_f32(is_set, ones, neg_ones);
                    svst1_f32(pg, &output[i], res);

                    i += svcntw();
                    pg = svwhilelt_b32(i, dimension);
                }
                return output;
            }
#endif

            // Dequantize Binary to FP32
            inline std::vector<float> dequantize(const uint8_t* buffer, size_t dimension) {
#if defined(USE_AVX512)
                return dequantize_avx512(buffer, dimension);
#elif defined(USE_AVX2)
                return dequantize_avx2(buffer, dimension);
#elif defined(USE_SVE2)
                return dequantize_sve(buffer, dimension);
#elif defined(USE_NEON)
                return dequantize_neon(buffer, dimension);
#else
                std::vector<float> output(dimension);
                const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(buffer);

                for(size_t i = 0; i < dimension; ++i) {
                    size_t chunk_idx = i / 64;
                    size_t bit_idx = i % 64;
                    if(data_ptr[chunk_idx] & (1ULL << bit_idx)) {
                        output[i] = 1.0f;
                    } else {
                        output[i] = -1.0f;
                    }
                }

                return output;
#endif
            }

            // Hamming distance implementation
            inline float Hamming(const void* v1, const void* v2, const void* params) {
                // params is expected to be a pointer to a struct where the first member is size_t
                // dim e.g. hnswlib::DistParams
                const size_t dim = *static_cast<const size_t*>(params);

                const uint64_t* p1 = static_cast<const uint64_t*>(v1);
                const uint64_t* p2 = static_cast<const uint64_t*>(v2);

                size_t num_uint64 = (dim + 63) / 64;
                float dist = 0;

                size_t i = 0;

#if defined(USE_AVX512)
                __m512i acc = _mm512_setzero_si512();

                for(; i + 8 <= num_uint64; i += 8) {
                    __m512i d1 = _mm512_loadu_si512((const __m512i*)&p1[i]);
                    __m512i d2 = _mm512_loadu_si512((const __m512i*)&p2[i]);
                    __m512i x = _mm512_xor_si512(d1, d2);
                    __m512i p = _mm512_popcnt_epi64(x);
                    acc = _mm512_add_epi64(acc, p);
                }

                // Handle remaining elements (1..7) using masking
                if(i < num_uint64) {
                    __mmask8 mask = (__mmask8)((1 << (num_uint64 - i)) - 1);
                    __m512i d1 = _mm512_maskz_loadu_epi64(mask, &p1[i]);
                    __m512i d2 = _mm512_maskz_loadu_epi64(mask, &p2[i]);
                    __m512i x = _mm512_xor_si512(d1, d2);
                    __m512i p = _mm512_popcnt_epi64(x);
                    acc = _mm512_add_epi64(acc, p);
                    i = num_uint64;  // skip scalar fallback
                }

                dist += _mm512_reduce_add_epi64(acc);

#elif defined(USE_AVX2)
                // AVX2 implementation using PSHUFB for population count
                __m256i mask_low = _mm256_set1_epi8(0x0F);
                __m256i lookup = _mm256_setr_epi8(0,
                                                  1,
                                                  1,
                                                  2,
                                                  1,
                                                  2,
                                                  2,
                                                  3,
                                                  1,
                                                  2,
                                                  2,
                                                  3,
                                                  2,
                                                  3,
                                                  3,
                                                  4,
                                                  0,
                                                  1,
                                                  1,
                                                  2,
                                                  1,
                                                  2,
                                                  2,
                                                  3,
                                                  1,
                                                  2,
                                                  2,
                                                  3,
                                                  2,
                                                  3,
                                                  3,
                                                  4);

                __m256i acc = _mm256_setzero_si256();

                for(; i + 4 <= num_uint64; i += 4) {
                    __m256i d1 = _mm256_loadu_si256((const __m256i*)&p1[i]);
                    __m256i d2 = _mm256_loadu_si256((const __m256i*)&p2[i]);
                    __m256i x = _mm256_xor_si256(d1, d2);

                    __m256i low = _mm256_and_si256(x, mask_low);
                    __m256i high = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask_low);

                    __m256i pop_low = _mm256_shuffle_epi8(lookup, low);
                    __m256i pop_high = _mm256_shuffle_epi8(lookup, high);

                    __m256i pop = _mm256_add_epi8(pop_low, pop_high);
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pop, _mm256_setzero_si256()));
                }

                uint64_t tmp[4];
                _mm256_storeu_si256((__m256i*)tmp, acc);
                dist += tmp[0] + tmp[1] + tmp[2] + tmp[3];

#elif defined(USE_SVE2)
                // SVE2 has population count instructions
                svuint64_t acc0 = svdup_u64(0);
                svuint64_t acc1 = svdup_u64(0);
                svuint64_t acc2 = svdup_u64(0);
                svuint64_t acc3 = svdup_u64(0);

                uint64_t num_elements = svcnth();  // Number of 64-bit elements (2 or 4)
                size_t unroll_stride = num_elements * 4;
                svbool_t pg_all = svptrue_b64();

                // Main unrolled loop
                for(; i + unroll_stride <= num_uint64; i += unroll_stride) {
                    svuint64_t d1_0 = svld1_u64(pg_all, &p1[i]);
                    svuint64_t d2_0 = svld1_u64(pg_all, &p2[i]);
                    svuint64_t x0 = sveor_u64_x(pg_all, d1_0, d2_0);
                    svuint64_t pop0 = svcnt_u64_x(pg_all, x0);
                    acc0 = svadd_u64_x(pg_all, acc0, pop0);

                    svuint64_t d1_1 = svld1_u64(pg_all, &p1[i + num_elements]);
                    svuint64_t d2_1 = svld1_u64(pg_all, &p2[i + num_elements]);
                    svuint64_t x1 = sveor_u64_x(pg_all, d1_1, d2_1);
                    svuint64_t pop1 = svcnt_u64_x(pg_all, x1);
                    acc1 = svadd_u64_x(pg_all, acc1, pop1);

                    svuint64_t d1_2 = svld1_u64(pg_all, &p1[i + 2 * num_elements]);
                    svuint64_t d2_2 = svld1_u64(pg_all, &p2[i + 2 * num_elements]);
                    svuint64_t x2 = sveor_u64_x(pg_all, d1_2, d2_2);
                    svuint64_t pop2 = svcnt_u64_x(pg_all, x2);
                    acc2 = svadd_u64_x(pg_all, acc2, pop2);

                    svuint64_t d1_3 = svld1_u64(pg_all, &p1[i + 3 * num_elements]);
                    svuint64_t d2_3 = svld1_u64(pg_all, &p2[i + 3 * num_elements]);
                    svuint64_t x3 = sveor_u64_x(pg_all, d1_3, d2_3);
                    svuint64_t pop3 = svcnt_u64_x(pg_all, x3);
                    acc3 = svadd_u64_x(pg_all, acc3, pop3);
                }

                svuint64_t acc = svadd_u64_x(pg_all, acc0, acc1);
                acc = svadd_u64_x(pg_all, acc, acc2);
                acc = svadd_u64_x(pg_all, acc, acc3);

                svbool_t pg = svwhilelt_b64(i, num_uint64);

                while(svptest_any(svptrue_b64(), pg)) {
                    svuint64_t d1 = svld1_u64(pg, &p1[i]);
                    svuint64_t d2 = svld1_u64(pg, &p2[i]);
                    svuint64_t x = sveor_u64_x(pg, d1, d2);

                    // CNT instruction counts bits in each element
                    svuint64_t p = svcnt_u64_x(pg, x);

                    // Critical fix: Use svadd_u64_m (merge) to preserve inactive lanes of acc
                    acc = svadd_u64_m(pg, acc, p);

                    i += svcntd();
                    pg = svwhilelt_b64(i, num_uint64);
                }
                dist += svaddv_u64(svptrue_b64(), acc);

#elif defined(USE_NEON)
                // NEON optimization
                // Unrolled 4x (process 8 uint64s = 64 bytes per iter)
                // Use 4 accumulators to break dependency chains

                uint16x8_t acc1 = vdupq_n_u16(0);
                uint16x8_t acc2 = vdupq_n_u16(0);
                uint16x8_t acc3 = vdupq_n_u16(0);
                uint16x8_t acc4 = vdupq_n_u16(0);

                for(; i + 7 < num_uint64; i += 8) {
                    // Load 4x 128-bit vectors (8 uint64s)
                    uint8x16_t d1 = veorq_u8(vld1q_u8((const uint8_t*)&p1[i]),
                                             vld1q_u8((const uint8_t*)&p2[i]));
                    uint8x16_t d2 = veorq_u8(vld1q_u8((const uint8_t*)&p1[i + 2]),
                                             vld1q_u8((const uint8_t*)&p2[i + 2]));
                    uint8x16_t d3 = veorq_u8(vld1q_u8((const uint8_t*)&p1[i + 4]),
                                             vld1q_u8((const uint8_t*)&p2[i + 4]));
                    uint8x16_t d4 = veorq_u8(vld1q_u8((const uint8_t*)&p1[i + 6]),
                                             vld1q_u8((const uint8_t*)&p2[i + 6]));

                    // Count bits (result is 8-bit counts)
                    uint8x16_t c1 = vcntq_u8(d1);
                    uint8x16_t c2 = vcntq_u8(d2);
                    uint8x16_t c3 = vcntq_u8(d3);
                    uint8x16_t c4 = vcntq_u8(d4);

                    // Accumulate pairwise into 16-bit accumulator
                    acc1 = vpadalq_u8(acc1, c1);
                    acc2 = vpadalq_u8(acc2, c2);
                    acc3 = vpadalq_u8(acc3, c3);
                    acc4 = vpadalq_u8(acc4, c4);
                }

                // Horizontal sum
                acc1 = vaddq_u16(acc1, acc2);
                acc3 = vaddq_u16(acc3, acc4);
                acc1 = vaddq_u16(acc1, acc3);

                // Handle remaining 128-bit chunks (2 uint64s)
                // Continue accumulating into acc1 to avoid scalar dependency chain
                for(; i + 1 < num_uint64; i += 2) {
                    uint8x16_t d = veorq_u8(vld1q_u8((const uint8_t*)&p1[i]),
                                            vld1q_u8((const uint8_t*)&p2[i]));
                    // Count bits in each byte (returns 8-bit counts)
                    uint8x16_t c = vcntq_u8(d);
                    // Accumulate into vector
                    acc1 = vpadalq_u8(acc1, c);
                }

                dist += vaddlvq_u16(acc1);
#endif

                // Scalar fallback / cleanup
                for(; i < num_uint64; ++i) {
                    dist += __builtin_popcountll(p1[i] ^ p2[i]);
                }

                return dist;
            }

            // Wrappers
            inline float L2Sqr(const void* v1, const void* v2, const void* params) {
                return Hamming(v1, v2, params);
            }

            inline float InnerProduct(const void* v1, const void* v2, const void* params) {
                return Hamming(v1, v2, params);
            }

            inline float Cosine(const void* v1, const void* v2, const void* params) {
                return Hamming(v1, v2, params);
            }

            // Similarity functions (higher is better)
            inline float HammingSim(const void* v1, const void* v2, const void* params) {
                const size_t dim = *static_cast<const size_t*>(params);
                return dim - Hamming(v1, v2, params);
            }

            inline float L2SqrSim(const void* v1, const void* v2, const void* params) {
                return HammingSim(v1, v2, params);
            }

            inline float InnerProductSim(const void* v1, const void* v2, const void* params) {
                return HammingSim(v1, v2, params);
            }

            inline float CosineSim(const void* v1, const void* v2, const void* params) {
                return HammingSim(v1, v2, params);
            }

            static std::vector<uint8_t> quantize_to_int8(const void* in, size_t dim) {
                throw std::runtime_error("Binary to Int8 direct quantization not implemented");
            }

        }  // namespace binary

        class BinaryQuantizer : public Quantizer {
        public:
            std::string name() const override { return "binary"; }
            QuantizationLevel level() const override { return QuantizationLevel::BINARY; }

            QuantizerDispatch getDispatch() const override {
                QuantizerDispatch d;
                d.dist_l2 = &binary::L2Sqr;
                d.dist_ip = &binary::InnerProduct;
                d.dist_cosine = &binary::Cosine;
                d.sim_l2 = &binary::L2SqrSim;
                d.sim_ip = &binary::InnerProductSim;
                d.sim_cosine = &binary::CosineSim;
                d.quantize = &binary::quantize;
                d.dequantize = &binary::dequantize;
                d.quantize_to_int8 = &binary::quantize_to_int8;
                d.get_storage_size = &binary::get_storage_size;
                d.extract_inv_scale = &binary::extract_inv_scale;
                return d;
            }
        };

        // Register BINARY
        static RegisterQuantizer reg_binary(QuantizationLevel::BINARY,
                                            "binary",
                                            std::make_shared<BinaryQuantizer>());

    }  // namespace quant
}  // namespace ndd
