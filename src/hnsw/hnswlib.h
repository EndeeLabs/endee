#pragma once

#include "../core/types.hpp"

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
#ifndef HNSWLIB_ERR_OVERRIDE
#    define HNSWERR std::cerr
#else
#    define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

#ifndef NO_MANUAL_VECTORIZATION
#    if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#        define USE_SSE
#        ifdef __AVX__
#            define USE_AVX
#        endif
#    endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#    ifdef _MSC_VER
#        include <intrin.h>
#        include <stdexcept>
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#    else
#        include <x86intrin.h>
#        include <cpuid.h>
#        include <stdint.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#    endif

#    if defined(__AVX512F__) || defined(__AVX512FP16__) || defined(__AVX512BW__)
#        include <immintrin.h>
#    endif

#    if defined(__GNUC__)
#        define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#        define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#    else
#        define PORTABLE_ALIGN32 __declspec(align(32))
#        define PORTABLE_ALIGN64 __declspec(align(64))
#    endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#    define _XCR_XFEATURE_ENABLED_MASK 0

static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if(nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if(osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

static bool AVX512Capable() {
    if(!AVXCapable()) {
        return false;
    }

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if(nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if(osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include "../quant/common.hpp"
#include "../core/types.hpp"

namespace hnswlib {
    typedef uint8_t SpaceType;
    static const SpaceType L2_SPACE = 0;
    static const SpaceType IP_SPACE = 1;
    static const SpaceType COSINE_SPACE = 2;

    struct DistParams {
        size_t dim;
        uint8_t quant_level;
    };

    inline SpaceType getSpaceType(const std::string& space_type_str) {
        if(space_type_str == "l2") {
            return L2_SPACE;
        }
        if(space_type_str == "ip") {
            return IP_SPACE;
        }
        if(space_type_str == "cosine") {
            return COSINE_SPACE;
        }
        throw std::runtime_error("Unknown space type: " + space_type_str);
    }

    inline std::string getSpaceTypeString(SpaceType space_type) {
        switch(space_type) {
            case L2_SPACE:
                return "l2";
            case IP_SPACE:
                return "ip";
            case COSINE_SPACE:
                return "cosine";
            default:
                throw std::runtime_error("Unknown space type: " + std::to_string(space_type));
        }
    }

    // Type alias for labels
    using idInt = ndd::idInt;
    // Type alias for storing internal IDs
    using idhInt = ndd::idhInt;
    // For storing various flags
    typedef uint32_t flagInt;
    // For storing level of the node
    typedef uint32_t levelInt;
    // This can be extended to store state for filtering (e.g. from a std::set)
    class BaseFilterFunctor {
    public:
        virtual bool operator()(idInt id) { return true; }
        virtual ~BaseFilterFunctor() {};
    };

    template <typename dist_t> class BaseSearchStopCondition {
    public:
        virtual void add_point_to_result(idInt label, const void* datapoint, dist_t dist) = 0;

        virtual void
        remove_point_from_result(ndd::idInt label, const void* datapoint, dist_t dist) = 0;

        virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

        virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

        virtual bool should_remove_extra() = 0;

        virtual void filter_results(std::vector<std::pair<dist_t, ndd::idInt>>& candidates) = 0;

        virtual ~BaseSearchStopCondition() {}
    };

    template <typename T> class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) { return p1.first > p2.first; }
    };

    template <typename T> static void writeBinaryPOD(std::ostream& out, const T& podRef) {
        out.write((char*)&podRef, sizeof(T));
    }

    template <typename T> static void readBinaryPOD(std::istream& in, T& podRef) {
        in.read((char*)&podRef, sizeof(T));
    }

    template <typename MTYPE> using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

    template <typename MTYPE> using SIMFUNC = MTYPE (*)(const void*, const void*, const void*);

    template <typename MTYPE> class SpaceInterface {
    public:
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual SIMFUNC<MTYPE> get_sim_func() = 0;

        virtual void* get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template <typename dist_t> class AlgorithmInterface {
    public:
        //virtual void addPoint(const void *datapoint, ndd::idInt label) = 0;

        virtual std::vector<std::pair<dist_t, ndd::idInt>>
        searchKnn(const void*, size_t, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

        virtual void saveIndex(const std::string& location) = 0;
        virtual ~AlgorithmInterface() {}
    };

}  // namespace hnswlib

#include "../core/space.hpp"

namespace hnswlib {
    template <typename dist_t>
    static SpaceInterface<dist_t>*
    createSpace(SpaceType space_type, size_t dim, ndd::quant::QuantizationLevel quant_level) {
        return new UnifiedSpace(space_type, dim, quant_level);
    }
}  //namespace hnswlib
//stop_condition.h is creating issue with AVX512
//#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
