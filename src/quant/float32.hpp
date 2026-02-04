#pragma once
#include "../hnsw/hnswlib.h"
#include "../quant/common.hpp"
#include "int8d.hpp"
#include <vector>
#include <cmath>
#include <cstring>

namespace hnswlib {
    namespace quant {
        namespace float32 {

            // =============================================================================
            // QUANTIZATION / DEQUANTIZATION
            // =============================================================================

            static std::vector<uint8_t> quantize_to_int8(const void* in, size_t dim) {
                const float* f_in = static_cast<const float*>(in);
                std::vector<float> input(f_in, f_in + dim);
#if defined(USE_SVE2)
                return ndd::quant::int8d::quantize_vector_fp32_to_int8_buffer_sve(input);
#elif defined(USE_AVX512)
                return ndd::quant::int8d::quantize_vector_fp32_to_int8_buffer_avx512(input);
#elif defined(USE_AVX2)
                return ndd::quant::int8d::quantize_vector_fp32_to_int8_buffer_avx2(input);
#elif defined(USE_NEON)
                return ndd::quant::int8d::quantize_vector_fp32_to_int8_buffer_neon(input);
#else
                return ndd::quant::int8d::quantize_vector_fp32_to_int8_buffer(input);
#endif
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                std::vector<uint8_t> result(input.size() * sizeof(float));
                std::memcpy(result.data(), input.data(), result.size());
                return result;
            }

            inline std::vector<float> dequantize(const uint8_t* buffer, size_t dimension) {
                std::vector<float> result(dimension);
                std::memcpy(result.data(), buffer, dimension * sizeof(float));
                return result;
            }

            inline float extract_inv_scale(const uint8_t* in, size_t dim) {
                return 1.0f;
            }

            // =============================================================================
            // DISTANCE IMPLEMENTATIONS
            // =============================================================================

            static float L2SqrScalar(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);
                float res = 0;
                for(size_t i = 0; i < qty; i++) {
                    float diff = vec1[i] - vec2[i];
                    res += diff * diff;
                }
                return res;
            }

            static float InnerProductScalar(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);
                float res = 0;
                for(size_t i = 0; i < qty; i++) {
                    res += vec1[i] * vec2[i];
                }
                return res;
            }

#if defined(USE_NEON)
            static float L2SqrNEON(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                float32x4_t sum1 = vdupq_n_f32(0);
                float32x4_t sum2 = vdupq_n_f32(0);
                float32x4_t sum3 = vdupq_n_f32(0);
                float32x4_t sum4 = vdupq_n_f32(0);

                size_t qty16 = qty >> 4;
                const float* pEnd1 = vec1 + (qty16 << 4);

                while(vec1 < pEnd1) {
                    float32x4_t v1_1 = vld1q_f32(vec1);
                    float32x4_t v2_1 = vld1q_f32(vec2);
                    float32x4_t diff1 = vsubq_f32(v1_1, v2_1);
                    sum1 = vfmaq_f32(sum1, diff1, diff1);

                    float32x4_t v1_2 = vld1q_f32(vec1 + 4);
                    float32x4_t v2_2 = vld1q_f32(vec2 + 4);
                    float32x4_t diff2 = vsubq_f32(v1_2, v2_2);
                    sum2 = vfmaq_f32(sum2, diff2, diff2);

                    float32x4_t v1_3 = vld1q_f32(vec1 + 8);
                    float32x4_t v2_3 = vld1q_f32(vec2 + 8);
                    float32x4_t diff3 = vsubq_f32(v1_3, v2_3);
                    sum3 = vfmaq_f32(sum3, diff3, diff3);

                    float32x4_t v1_4 = vld1q_f32(vec1 + 12);
                    float32x4_t v2_4 = vld1q_f32(vec2 + 12);
                    float32x4_t diff4 = vsubq_f32(v1_4, v2_4);
                    sum4 = vfmaq_f32(sum4, diff4, diff4);

                    vec1 += 16;
                    vec2 += 16;
                }

                float32x4_t sum = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
                float res = vaddvq_f32(sum);

                // Handle remaining elements
                size_t qty_left = qty - (qty16 << 4);
                for(size_t i = 0; i < qty_left; i++) {
                    float diff = vec1[i] - vec2[i];
                    res += diff * diff;
                }

                return res;
            }

            static float InnerProductNEON(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                float32x4_t sum1 = vdupq_n_f32(0);
                float32x4_t sum2 = vdupq_n_f32(0);
                float32x4_t sum3 = vdupq_n_f32(0);
                float32x4_t sum4 = vdupq_n_f32(0);

                size_t qty16 = qty >> 4;
                const float* pEnd1 = vec1 + (qty16 << 4);

                while(vec1 < pEnd1) {
                    float32x4_t v1_1 = vld1q_f32(vec1);
                    float32x4_t v2_1 = vld1q_f32(vec2);
                    sum1 = vfmaq_f32(sum1, v1_1, v2_1);

                    float32x4_t v1_2 = vld1q_f32(vec1 + 4);
                    float32x4_t v2_2 = vld1q_f32(vec2 + 4);
                    sum2 = vfmaq_f32(sum2, v1_2, v2_2);

                    float32x4_t v1_3 = vld1q_f32(vec1 + 8);
                    float32x4_t v2_3 = vld1q_f32(vec2 + 8);
                    sum3 = vfmaq_f32(sum3, v1_3, v2_3);

                    float32x4_t v1_4 = vld1q_f32(vec1 + 12);
                    float32x4_t v2_4 = vld1q_f32(vec2 + 12);
                    sum4 = vfmaq_f32(sum4, v1_4, v2_4);

                    vec1 += 16;
                    vec2 += 16;
                }

                float32x4_t sum = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
                float res = vaddvq_f32(sum);

                // Handle remaining elements
                size_t qty_left = qty - (qty16 << 4);
                for(size_t i = 0; i < qty_left; i++) {
                    res += vec1[i] * vec2[i];
                }

                return res;
            }
#endif

#if defined(USE_AVX512)
            static float L2SqrAVX512(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                size_t qty16 = qty >> 4;
                const float* pEnd1 = vec1 + (qty16 << 4);

                __m512 sum = _mm512_setzero_ps();

                while(vec1 < pEnd1) {
                    __m512 v1 = _mm512_loadu_ps(vec1);
                    __m512 v2 = _mm512_loadu_ps(vec2);
                    __m512 diff = _mm512_sub_ps(v1, v2);
                    sum = _mm512_fmadd_ps(diff, diff, sum);
                    vec1 += 16;
                    vec2 += 16;
                }

                float res = _mm512_reduce_add_ps(sum);

                size_t qty_left = qty - (qty16 << 4);
                for(size_t i = 0; i < qty_left; i++) {
                    float diff = vec1[i] - vec2[i];
                    res += diff * diff;
                }

                return res;
            }

            static float InnerProductAVX512(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                size_t qty16 = qty >> 4;
                const float* pEnd1 = vec1 + (qty16 << 4);

                __m512 sum = _mm512_setzero_ps();

                while(vec1 < pEnd1) {
                    __m512 v1 = _mm512_loadu_ps(vec1);
                    __m512 v2 = _mm512_loadu_ps(vec2);
                    sum = _mm512_fmadd_ps(v1, v2, sum);
                    vec1 += 16;
                    vec2 += 16;
                }

                float res = _mm512_reduce_add_ps(sum);

                size_t qty_left = qty - (qty16 << 4);
                for(size_t i = 0; i < qty_left; i++) {
                    res += vec1[i] * vec2[i];
                }

                return res;
            }
#endif

#if defined(USE_AVX2)
            static float L2SqrAVX2(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                size_t qty8 = qty >> 3;
                const float* pEnd1 = vec1 + (qty8 << 3);

                __m256 sum = _mm256_setzero_ps();

                while(vec1 < pEnd1) {
                    __m256 v1 = _mm256_loadu_ps(vec1);
                    __m256 v2 = _mm256_loadu_ps(vec2);
                    __m256 diff = _mm256_sub_ps(v1, v2);
#    if defined(__FMA__)
                    sum = _mm256_fmadd_ps(diff, diff, sum);
#    else
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
#    endif
                    vec1 += 8;
                    vec2 += 8;
                }

                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);

                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                float res = _mm_cvtss_f32(sum128);

                size_t qty_left = qty - (qty8 << 3);
                for(size_t i = 0; i < qty_left; i++) {
                    float diff = vec1[i] - vec2[i];
                    res += diff * diff;
                }

                return res;
            }

            static float InnerProductAVX2(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                size_t qty8 = qty >> 3;
                const float* pEnd1 = vec1 + (qty8 << 3);

                __m256 sum = _mm256_setzero_ps();

                while(vec1 < pEnd1) {
                    __m256 v1 = _mm256_loadu_ps(vec1);
                    __m256 v2 = _mm256_loadu_ps(vec2);
#    if defined(__FMA__)
                    sum = _mm256_fmadd_ps(v1, v2, sum);
#    else
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
#    endif
                    vec1 += 8;
                    vec2 += 8;
                }

                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);

                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                float res = _mm_cvtss_f32(sum128);

                size_t qty_left = qty - (qty8 << 3);
                for(size_t i = 0; i < qty_left; i++) {
                    res += vec1[i] * vec2[i];
                }

                return res;
            }
#endif

#if defined(USE_SVE2)
            static float L2SqrSVE(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                svfloat32_t sum1 = svdup_f32(0.0f);
                svfloat32_t sum2 = svdup_f32(0.0f);
                svfloat32_t sum3 = svdup_f32(0.0f);
                svfloat32_t sum4 = svdup_f32(0.0f);

                size_t i = 0;
                uint64_t num_elements = svcntw();
                size_t unroll_stride = num_elements * 4;

                svbool_t pg_all = svptrue_b32();

                // Main unrolled loop
                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svfloat32_t v1_1 = svld1_f32(pg_all, vec1 + i);
                    svfloat32_t v2_1 = svld1_f32(pg_all, vec2 + i);
                    svfloat32_t diff1 = svsub_f32_x(pg_all, v1_1, v2_1);
                    sum1 = svmla_f32_x(pg_all, sum1, diff1, diff1);

                    svfloat32_t v1_2 = svld1_f32(pg_all, vec1 + i + num_elements);
                    svfloat32_t v2_2 = svld1_f32(pg_all, vec2 + i + num_elements);
                    svfloat32_t diff2 = svsub_f32_x(pg_all, v1_2, v2_2);
                    sum2 = svmla_f32_x(pg_all, sum2, diff2, diff2);

                    svfloat32_t v1_3 = svld1_f32(pg_all, vec1 + i + 2 * num_elements);
                    svfloat32_t v2_3 = svld1_f32(pg_all, vec2 + i + 2 * num_elements);
                    svfloat32_t diff3 = svsub_f32_x(pg_all, v1_3, v2_3);
                    sum3 = svmla_f32_x(pg_all, sum3, diff3, diff3);

                    svfloat32_t v1_4 = svld1_f32(pg_all, vec1 + i + 3 * num_elements);
                    svfloat32_t v2_4 = svld1_f32(pg_all, vec2 + i + 3 * num_elements);
                    svfloat32_t diff4 = svsub_f32_x(pg_all, v1_4, v2_4);
                    sum4 = svmla_f32_x(pg_all, sum4, diff4, diff4);
                }

                // Reduce sums
                svfloat32_t sum = svadd_f32_x(pg_all, sum1, sum2);
                sum = svadd_f32_x(pg_all, sum, sum3);
                sum = svadd_f32_x(pg_all, sum, sum4);

                // Handle remainder
                svbool_t pg = svwhilelt_b32(i, qty);
                while(svptest_any(svptrue_b32(), pg)) {
                    svfloat32_t v1 = svld1_f32(pg, vec1 + i);
                    svfloat32_t v2 = svld1_f32(pg, vec2 + i);
                    svfloat32_t diff = svsub_f32_x(pg, v1, v2);
                    sum = svmla_f32_x(pg, sum, diff, diff);

                    i += num_elements;
                    pg = svwhilelt_b32(i, qty);
                }

                return svaddv_f32(svptrue_b32(), sum);
            }

            static float InnerProductSVE(const void* pVect1, const void* pVect2, size_t qty) {
                const float* vec1 = reinterpret_cast<const float*>(pVect1);
                const float* vec2 = reinterpret_cast<const float*>(pVect2);

                svfloat32_t sum1 = svdup_f32(0.0f);
                svfloat32_t sum2 = svdup_f32(0.0f);
                svfloat32_t sum3 = svdup_f32(0.0f);
                svfloat32_t sum4 = svdup_f32(0.0f);

                size_t i = 0;
                uint64_t num_elements = svcntw();
                size_t unroll_stride = num_elements * 4;

                svbool_t pg_all = svptrue_b32();

                // Main unrolled loop
                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svfloat32_t v1_1 = svld1_f32(pg_all, vec1 + i);
                    svfloat32_t v2_1 = svld1_f32(pg_all, vec2 + i);
                    sum1 = svmla_f32_x(pg_all, sum1, v1_1, v2_1);

                    svfloat32_t v1_2 = svld1_f32(pg_all, vec1 + i + num_elements);
                    svfloat32_t v2_2 = svld1_f32(pg_all, vec2 + i + num_elements);
                    sum2 = svmla_f32_x(pg_all, sum2, v1_2, v2_2);

                    svfloat32_t v1_3 = svld1_f32(pg_all, vec1 + i + 2 * num_elements);
                    svfloat32_t v2_3 = svld1_f32(pg_all, vec2 + i + 2 * num_elements);
                    sum3 = svmla_f32_x(pg_all, sum3, v1_3, v2_3);

                    svfloat32_t v1_4 = svld1_f32(pg_all, vec1 + i + 3 * num_elements);
                    svfloat32_t v2_4 = svld1_f32(pg_all, vec2 + i + 3 * num_elements);
                    sum4 = svmla_f32_x(pg_all, sum4, v1_4, v2_4);
                }

                // Reduce sums
                svfloat32_t sum = svadd_f32_x(pg_all, sum1, sum2);
                sum = svadd_f32_x(pg_all, sum, sum3);
                sum = svadd_f32_x(pg_all, sum, sum4);

                // Handle remainder
                svbool_t pg = svwhilelt_b32(i, qty);
                while(svptest_any(svptrue_b32(), pg)) {
                    svfloat32_t v1 = svld1_f32(pg, vec1 + i);
                    svfloat32_t v2 = svld1_f32(pg, vec2 + i);
                    sum = svmla_f32_x(pg, sum, v1, v2);

                    i += num_elements;
                    pg = svwhilelt_b32(i, qty);
                }

                return svaddv_f32(svptrue_b32(), sum);
            }
#endif

            static float L2Sqr(const void* pVect1, const void* pVect2, size_t qty) {
#if defined(USE_AVX512)
                return L2SqrAVX512(pVect1, pVect2, qty);
#elif defined(USE_SVE2)
                return L2SqrSVE(pVect1, pVect2, qty);
#elif defined(USE_AVX2)
                return L2SqrAVX2(pVect1, pVect2, qty);
#elif defined(USE_NEON)
                return L2SqrNEON(pVect1, pVect2, qty);
#else
                return L2SqrScalar(pVect1, pVect2, qty);
#endif
            }

            static float InnerProduct(const void* pVect1, const void* pVect2, size_t qty) {
#if defined(USE_AVX512)
                return InnerProductAVX512(pVect1, pVect2, qty);
#elif defined(USE_SVE2)
                return InnerProductSVE(pVect1, pVect2, qty);
#elif defined(USE_AVX2)
                return InnerProductAVX2(pVect1, pVect2, qty);
#elif defined(USE_NEON)
                return InnerProductNEON(pVect1, pVect2, qty);
#else
                return InnerProductScalar(pVect1, pVect2, qty);
#endif
            }

            static float
            L2SqrDistance(const void* pVect1, const void* pVect2, const void* params_ptr) {
                const DistParams* params = reinterpret_cast<const DistParams*>(params_ptr);
                return L2Sqr(pVect1, pVect2, params->dim);
            }

            static float L2SqrSim(const void* pVect1, const void* pVect2, const void* params_ptr) {
                return -L2SqrDistance(pVect1, pVect2, params_ptr);
            }

            static float
            InnerProductSim(const void* pVect1, const void* pVect2, const void* params_ptr) {
                const DistParams* params = reinterpret_cast<const DistParams*>(params_ptr);
                return InnerProduct(pVect1, pVect2, params->dim);
            }

            static float CosineSim(const void* pVect1, const void* pVect2, const void* params_ptr) {
                return InnerProductSim(pVect1, pVect2, params_ptr);
            }

            static float
            InnerProductDistance(const void* pVect1, const void* pVect2, const void* params_ptr) {
                const DistParams* params = reinterpret_cast<const DistParams*>(params_ptr);
                return 1.0f - InnerProduct(pVect1, pVect2, params->dim);
            }

            static float
            CosineDistance(const void* pVect1, const void* pVect2, const void* params_ptr) {
                // For normalized vectors, Cosine distance is same as Inner Product distance
                return InnerProductDistance(pVect1, pVect2, params_ptr);
            }

        }  //namespace float32
    }  // namespace quant

    class FP32Space : public SpaceInterface<float> {
    private:
        size_t dim_;
        size_t data_size_;
        DistParams dist_params_;
        SpaceType space_type_;

    public:
        FP32Space(SpaceType space_type, size_t dim) :
            dim_(dim),
            space_type_(space_type) {
            data_size_ = dim * sizeof(float);
            dist_params_.dim = dim;
            dist_params_.quant_level = static_cast<uint8_t>(ndd::quant::QuantizationLevel::FP32);
        }

        size_t get_data_size() override { return data_size_; }

        DISTFUNC<float> get_dist_func() override {
            switch(space_type_) {
                case L2_SPACE:
                    return quant::float32::L2SqrDistance;
                case IP_SPACE:
                    return quant::float32::InnerProductDistance;
                case COSINE_SPACE:
                    return quant::float32::CosineDistance;
                default:
                    throw std::runtime_error("Unknown space type");
            }
        }

        SIMFUNC<float> get_sim_func() override {
            switch(space_type_) {
                case L2_SPACE:
                    return [](const void* v1, const void* v2, const void* param) -> float {
                        const DistParams* params = reinterpret_cast<const DistParams*>(param);
                        return -quant::float32::L2Sqr(
                                v1, v2, params->dim);  // Negative L2 for similarity
                    };
                case IP_SPACE:
                case COSINE_SPACE:
                    return [](const void* v1, const void* v2, const void* param) -> float {
                        const DistParams* params = reinterpret_cast<const DistParams*>(param);
                        return quant::float32::InnerProduct(v1, v2, params->dim);
                    };
                default:
                    throw std::runtime_error("Unknown space type");
            }
        }

        void* get_dist_func_param() override { return &dist_params_; }
    };

}  // namespace hnswlib

namespace ndd {
    namespace quant {

        class Float32Quantizer : public Quantizer {
        public:
            std::string name() const override { return "float32"; }
            QuantizationLevel level() const override { return QuantizationLevel::FP32; }

            QuantizerDispatch getDispatch() const override {
                QuantizerDispatch d;
                d.dist_l2 = &hnswlib::quant::float32::L2SqrDistance;
                d.dist_ip = &hnswlib::quant::float32::InnerProductDistance;
                d.dist_cosine = &hnswlib::quant::float32::CosineDistance;
                d.sim_l2 = &hnswlib::quant::float32::L2SqrSim;
                d.sim_ip = &hnswlib::quant::float32::InnerProductSim;
                d.sim_cosine = &hnswlib::quant::float32::CosineSim;
                d.quantize = &hnswlib::quant::float32::quantize;
                d.dequantize = &hnswlib::quant::float32::dequantize;
                d.quantize_to_int8 = &hnswlib::quant::float32::quantize_to_int8;
                d.get_storage_size = [](size_t dim) { return dim * sizeof(float); };
                d.extract_inv_scale = &hnswlib::quant::float32::extract_inv_scale;
                return d;
            }
        };

        // Register FP32
        static RegisterQuantizer
                reg_fp32(QuantizationLevel::FP32, "float32", std::make_shared<Float32Quantizer>());

    }  // namespace quant
}  // namespace ndd
