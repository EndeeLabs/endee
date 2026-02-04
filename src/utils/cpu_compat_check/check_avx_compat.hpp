#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

static const char* ERROR_CALLING_FUNC = "ERROR: should not be calling %s\n";
static const uint32_t ECX_OSXSAVE_BIT = 27;
static const uint32_t ECX_AVX_BIT = 28;

static const uint32_t CPUID_FEATURES_LEAF = 1;
static const uint32_t CPUID_EXT_FEATURES_LEAF = 7;
static const uint32_t CPUID_SUBLEAF_0 = 0;

static const uint32_t EBX_AVX2_BIT = 5;
static const uint32_t EBX_AVX512F_BIT = 16;
static const uint32_t EBX_AVX512BW_BIT = 30;
static const uint32_t ECX_AVX512VNNI_BIT = 11;
static const uint32_t ECX_AVX512VPOPCNTDQ_BIT = 14;
static const uint32_t EDX_AVX512FP16_BIT = 23;

#if defined(__aarch64__)

/**
 * Always return false if these functions are called
 */

int check_avx2_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

int check_avx512_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

int check_avx512_fp16_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

int check_avx512_vnni_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

int check_avx512_bw_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

int check_avx512_vpopcntdq_support(void) {
    printf(ERROR_CALLING_FUNC, __func__);
    return false;
}

#else  //X86
#    include <cpuid.h>
#    include <immintrin.h>

static void cpuid_ex(uint32_t leaf,
                     uint32_t subleaf,
                     uint32_t* eax,
                     uint32_t* ebx,
                     uint32_t* ecx,
                     uint32_t* edx) {
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
}

static uint64_t xgetbv0(void) {
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}

/**
 * //////////////////////////////////////////////////////////////////
 * CPU Type Checks
 * /////////////////////////////////////////////////////////////////
 */

/**
 * returns true if the CPU is intel, else false
 */
static int is_intel_cpu(void) {
    return true;

#    if 0
        uint32_t eax, ebx, ecx, edx;
        char vendor[13];
        cpuid_ex(0, 0, &eax, &ebx, &ecx, &edx);
        ((uint32_t*)vendor)[0] = ebx;
        ((uint32_t*)vendor)[1] = edx;
        ((uint32_t*)vendor)[2] = ecx;
        vendor[12] = 0;
        return strcmp(vendor, "GenuineIntel") == 0;
#    endif  //if 0
}

/**
 * //////////////////////////////////////////////////////////////////
 * OS Checks
 * /////////////////////////////////////////////////////////////////
 */

/**
 * True if OS supports AVX (XMM+YMM state enabled in XCR0), else false
 * Needed to safely execute AVX/AVX2 instructions.
 */
static int os_supports_avx(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid_ex(CPUID_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);

    const int osxsave = (ecx >> ECX_OSXSAVE_BIT) & 1;
    const int avx = (ecx >> ECX_AVX_BIT) & 1;
    if(!osxsave || !avx) {
        return 0;
    }

    uint64_t xcr0 = xgetbv0();
    // XCR0 bits: 1=XMM, 2=YMM must be enabled for AVX/AVX2
    const uint64_t needed = (1ULL << 1) | (1ULL << 2);
    return ((xcr0 & needed) == needed);
}

/**
 * true if OS supports AVX512 state, else false
 */
static int os_supports_avx512_state(void) {
    // Need OSXSAVE set and XCR0 enabling XMM/YMM and Opmask/ZMM state.
    uint32_t eax, ebx, ecx, edx;
    cpuid_ex(CPUID_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);

    const int osxsave = (ecx >> ECX_OSXSAVE_BIT) & 1;
    const int avx = (ecx >> ECX_AVX_BIT) & 1;
    if(!osxsave || !avx) {
        return 0;
    }

    uint64_t xcr0 = xgetbv0();
    // XCR0 bits: 1=XMM, 2=YMM, 5=opmask, 6=ZMM_hi256, 7=hi16_ZMM
    const uint64_t needed = (1ULL << 1) | (1ULL << 2) | (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    return ((xcr0 & needed) == needed);
}

/**
 * //////////////////////////////////////////////////////////////////
 * CPU Instruction Checks
 * /////////////////////////////////////////////////////////////////
 */

/**
 * True if CPU has AVX2 (hardware feature), else false
 */
static int cpu_has_avx2(void) {
    uint32_t eax, ebx, ecx, edx;

    // AVX2: CPUID.(EAX=7, ECX=0):EBX bit 5
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);
    return ((ebx >> EBX_AVX2_BIT) & 1);
}

/**
 * True if CPU has AVX-512F (base AVX-512), else false
 * This intentionally does NOT require AVX512_FP16.
 */
static int cpu_has_avx512f(void) {
    uint32_t eax, ebx, ecx, edx;

    // AVX-512F: CPUID.(EAX=7, ECX=0):EBX bit 16
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);
    return ((ebx >> EBX_AVX512F_BIT) & 1);
}

/**
 * True if CPU has AVX512f and fp16
 */
static int cpu_has_avx512f_and_fp16(void) {
    uint32_t eax, ebx, ecx, edx;

    // Leaf 7, subleaf 0
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);

    // AVX-512F: EBX bit 16
    int avx512f = (ebx >> EBX_AVX512F_BIT) & 1;
    if(!avx512f) {
        return 0;
    }

    // AVX-512 FP16: EDX bit 23  (NOT leaf 7 subleaf 1)
    int avx512fp16 = (edx >> EDX_AVX512FP16_BIT) & 1;

    return avx512fp16;
}

static int cpu_has_avx512vnni(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);
    return (ecx >> ECX_AVX512VNNI_BIT) & 1;
}

static int cpu_has_avx512bw(void) {
    uint32_t eax, ebx, ecx, edx;
    // AVX-512BW: CPUID.(EAX=7, ECX=0):EBX bit 30
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);
    return (ebx >> EBX_AVX512BW_BIT) & 1;
}

/**
 * True if CPU has AVX512 VPOPCNTDQ (vector population count for dword/qword)
 */
static int cpu_has_avx512vpopcntdq(void) {
    uint32_t eax, ebx, ecx, edx;
    // AVX-512 VPOPCNTDQ: CPUID.(EAX=7, ECX=0):ECX bit 14
    cpuid_ex(CPUID_EXT_FEATURES_LEAF, CPUID_SUBLEAF_0, &eax, &ebx, &ecx, &edx);
    return (ecx >> ECX_AVX512VPOPCNTDQ_BIT) & 1;
}

/**
 * //////////////////////////////////////////////////////////////////
 * One Instruction test
 * /////////////////////////////////////////////////////////////////
 */

#    if 0
/**
 * Optional: actually execute one AVX2 instruction (safe only if checks pass).
 * Keep it commented out like you did for FP16 if you don't want to run it.
 */
static void run_one_avx2_instruction(void) {
        // Simple AVX2 op: vpaddd ymm,ymm,ymm
        __m256i a = _mm256_setzero_si256();
        __m256i b = _mm256_setzero_si256();
        __m256i c = _mm256_add_epi32(a, b);
        (void)c;
}

/**
 * Optional: execute one plain AVX-512 instruction (no FP16).
 * NOTE: This function must only be called after cpu_has_avx512f() and
 * os_supports_avx512_state() pass, or it may SIGILL.
 */
static void run_one_avx512_instruction(void) {
        // Simple AVX-512F op: vpaddd zmm,zmm,zmm
        __m512i a = _mm512_setzero_si512();
        __m512i b = _mm512_setzero_si512();
        __m512i c = _mm512_add_epi32(a, b);
        (void)c;
}

/**
 * An actual AVX512-fp16 instruction to see if it works.
 * Will give illegal instruction if run on CPU without this support
 */
static void run_one_fp16_instruction(void) {
        // vaddph zmm,zmm,zmm  (adds packed FP16) as a sanity check.
        // Use zeroed vectors so result is stable.
        __m512h a = _mm512_setzero_ph();
        __m512h b = _mm512_setzero_ph();
        __m512h c = _mm512_add_ph(a, b);
        (void)c;
}

static void run_one_avx512vnni_instruction(void) {
        __m512i acc = _mm512_setzero_si512();
        __m512i a   = _mm512_setzero_si512();
        __m512i b   = _mm512_setzero_si512();
        // VPDPBUSD: dot-product of unsigned bytes * signed bytes accumulating into dwords
        __m512i c = _mm512_dpbusd_epi32(acc, a, b);
        (void)c;
}

static void run_one_avx512bw_instruction(void) {
        // VPABSB zmm,zmm  (requires AVX-512BW)
        __m512i a = _mm512_set1_epi8(-5);
        __m512i b = _mm512_abs_epi8(a);
        (void)b;
}

static void run_one_avx512vpopcntdq_instruction(void) {
    // VPOPCNTQ zmm,zmm  (requires AVX-512 VPOPCNTDQ)
    __m512i a = _mm512_set1_epi64(0x123456789ABCDEF0ULL);
    __m512i b = _mm512_popcnt_epi64(a);
    (void)b;
}
#    endif
/**
 * //////////////////////////////////////////////////////////////////
 * Top level functions to be called to test
 * /////////////////////////////////////////////////////////////////
 */

int check_avx2_support(void) {
    int ret = false;

    if(!cpu_has_avx2()) {
        printf("ERROR: AVX2: not supported by CPU.\n");
        goto exit;
    }

    if(!os_supports_avx()) {
        printf("ERROR: AVX2: CPU supports it, but OS/XSAVE state not enabled (XCR0 XMM/YMM).\n");
        goto exit;
    }

    //If you want, you can run a real AVX2 instruction here:
    // run_one_avx2_instruction();

    ret = true;
    printf("LOG: AVX2: supported and usable.\n");
exit:
    return ret;
}

/**
 * Returns true if AVX-512 is supported and usable (AVX-512F + OS state).
 * Should PASS on CPUs with AVX-512 but WITHOUT AVX512_FP16.
 */
int check_avx512_support(void) {
    int ret = false;

    if(!cpu_has_avx512f()) {
        printf("ERROR: AVX-512: not supported by CPU (missing AVX-512F).\n");
        goto exit;
    }

    if(!os_supports_avx512_state()) {
        printf("ERROR: AVX-512: CPU supports it, but OS/XSAVE state not enabled (XCR0).\n");
        goto exit;
    }

    // Optional runtime proof (safe after checks pass):
    // run_one_avx512_instruction();

    ret = true;
    printf("LOG: AVX-512: supported and usable.\n");
exit:
    return ret;
}

int check_avx512_fp16_support(void) {
    int ret = false;

    if(!is_intel_cpu()) {
        printf("ERROR: AVX-512 FP16: not supported on non-Intel CPUs.\n");
        goto exit;
    }

    if(!cpu_has_avx512f_and_fp16()) {
        printf("ERROR: AVX-512 FP16: not supported by CPU.\n");
        goto exit;
    }

    if(!os_supports_avx512_state()) {
        printf("ERROR: AVX-512 FP16: CPU supports it, but OS/XSAVE state not enabled (XCR0).\n");
        goto exit;
    }

    // If we got here, it should be safe to execute AVX-512 instructions.
    // run_one_fp16_instruction();

    ret = true;
    printf("LOG: AVX-512 FP16: supported and usable.\n");
exit:
    return ret;
}

int check_avx512_vnni_support(void) {
    int ret = false;

    if(!cpu_has_avx512f()) {
        printf("ERROR: AVX-512 VNNI: not supported (missing AVX-512F).\n");
        goto exit;
    }

    if(!cpu_has_avx512vnni()) {
        printf("ERROR: AVX-512 VNNI: not supported by CPU.\n");
        goto exit;
    }

    if(!os_supports_avx512_state()) {
        printf("ERROR: AVX-512 VNNI: CPU supports it, but OS AVX-512 state not enabled (XCR0).\n");
        goto exit;
    }

    // run_one_avx512vnni_instruction();

    ret = true;
    printf("LOG: AVX-512 VNNI: supported and usable.\n");

exit:
    return ret;
}

int check_avx512_bw_support(void) {
    int ret = false;

    if(!cpu_has_avx512f()) {
        printf("ERROR: AVX-512 BW: not supported (missing AVX-512F).\n");
        goto exit;
    }

    if(!cpu_has_avx512bw()) {
        printf("ERROR: AVX-512 BW: not supported by CPU.\n");
        goto exit;
    }

    if(!os_supports_avx512_state()) {
        printf("ERROR: AVX-512 BW: CPU supports it, but OS AVX-512 state not enabled (XCR0).\n");
        goto exit;
    }

    // run_one_avx512bw_instruction();

    ret = true;
    printf("LOG: AVX-512 BW: supported and usable.\n");

exit:
    return ret;
}

int check_avx512_vpopcntdq_support(void) {
    int ret = false;

    if(!cpu_has_avx512f()) {
        printf("ERROR: AVX-512 vpopcntdq: not supported (missing AVX-512F).\n");
        goto exit;
    }

    if(!cpu_has_avx512vpopcntdq()) {
        printf("ERROR: AVX-512 vpopcntdq: not supported by CPU.\n");
        goto exit;
    }

    if(!os_supports_avx512_state()) {
        printf("ERROR: AVX-512 vpopcntdq: CPU supports it, but OS AVX-512 state not enabled "
               "(XCR0).\n");
        goto exit;
    }

    // run_one_avx512vpopcntdq_instruction();

    ret = true;
    printf("LOG: AVX-512 vpopcntdq: supported and usable.\n");

exit:
    return ret;
}

#endif  //__aarch64__

/**
 * Supported CPUs: Intel Sapphire Rapids (4th Gen Xeon, 2023), Emerald Rapids, Meteor Lake
 * (P-cores), and AMD Zen 5.
 */

bool is_avx2_compatible() {
    return check_avx2_support();
}

bool is_avx512_compatible() {
    return check_avx2_support() && check_avx512_support() && check_avx512_fp16_support()
           && check_avx512_vnni_support() && check_avx512_bw_support()
           && check_avx512_vpopcntdq_support();
}

/*
int main(){
        //gcc test_avx.c -mavx512f -mavx512fp16 -mavx512vnni -mavx512bw -mavx512vpopcntdq
        if(is_avx512_compatible()){
                printf("YES\n");
        }else{
                printf("NO\n");
        }
        return 0;
}
*/
