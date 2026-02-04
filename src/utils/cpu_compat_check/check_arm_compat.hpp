// Runtime instruction probes for NEON and SVE2 with graceful SIGILL handling.

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#if !defined(__aarch64__)

/**
 * Always return false if these functions are called
 * when aarch64 is not defined while make
 */

int is_neon_compatible(void) {
    printf("ERROR: should not be calling %s\n", __func__);
    return false;
}

int is_sve2_compatible(void) {
    printf("ERROR: should not be calling %s\n", __func__);
    return false;
}

#else  //__aarch64__ is defined

#    include <signal.h>
#    include <setjmp.h>

static sigjmp_buf g_jmp;
static struct sigaction g_old_ill;

static void sigill_handler(int sig) {
    (void)sig;
    siglongjmp(g_jmp, 1);
}

// Run fn() and return 1 if it executed, 0 if it trapped SIGILL.
static int run_sigill_safe(void (*fn)(void)) {
    struct sigaction sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if(sigaction(SIGILL, &sa, &g_old_ill) != 0) {
        return 0;
    }

    if(sigsetjmp(g_jmp, 1) != 0) {
        sigaction(SIGILL, &g_old_ill, NULL);
        return 0;
    }

    fn();

    sigaction(SIGILL, &g_old_ill, NULL);
    return 1;
}

/* ---------------- NEON probe (always available on AArch64) ---------------- */
#    include <arm_neon.h>

static void neon_one_instruction(void) {
    // A simple NEON add: vaddq_u8
    uint8x16_t a = vdupq_n_u8(1);
    uint8x16_t b = vdupq_n_u8(2);
    uint8x16_t c = vaddq_u8(a, b);
    (void)c;
}

static int probe_neon(void) {
    return run_sigill_safe(neon_one_instruction);
}

/* ---------------- SVE2 probe (only if compiled with SVE2 enabled) ---------------- */
#    if defined(__ARM_FEATURE_SVE2)
#        include <arm_sve.h>

static void sve2_one_instruction(void) {
    // Use an SVE2-specific intrinsic (widening add long bottom).
    // This should map to an SVE2 instruction and SIGILL on non-SVE2 CPUs.
    svuint8_t a = svdup_u8(1);
    svuint8_t b = svdup_u8(2);
    svuint16_t c = svaddlb_u16(a, b);  // SVE2
    (void)c;
}

static int probe_sve2(void) {
    return run_sigill_safe(sve2_one_instruction);
}
#    else
// Not compiled for SVE2 => cannot execute SVE2 instructions from this binary.
static int probe_sve2(void) {
    return 0;
}
#    endif

//////////////////////////////////////////////////////////////////
// These are the top level functions. Only these should be called
//////////////////////////////////////////////////////////////////

int is_neon_compatible(void) {
    int neon_ok = probe_neon();

    if(neon_ok) {
        printf("LOG: NEON: supported and usable.\n");
    } else {
        printf("ERROR: NEON: NOT supported.\n");
    }
    return neon_ok;
}

int is_sve2_compatible(void) {
    int sve2_ok = probe_sve2();

    if(sve2_ok) {
        printf("LOG: SVE2: supported and usable.\n");
    } else {
        printf("ERROR: SVE2: NOT supported.\n");
    }

    return sve2_ok;
}

#endif  //__aarch64__
