// PTO Program: nn_RMSNorm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_RMSNorm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 896 bytes (0.9 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            320 bytes (35.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mean_sq              8x1        f32        32   [  3,   4]           -
//   mean_sq_eps          8x1        f32        32   [  4,   5]           <- mean_sq_sum
//   mean_sq_sum          8x1        f32        32   [  2,   3]           -
//   result               8x8        f32       256   [  6,   7]           <- x_squared
//   rms                  8x1        f32        32   [  5,   6]           <- mean_sq
//   x                    8x8        f32       256   [  0,   6]           -
//   x_squared            8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   mean_sq_eps reuses buffer of mean_sq_sum
//   rms reuses buffer of mean_sq
//   result reuses buffer of x_squared
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_RMSNorm(float* input, float* output) {
    float x[8][8];
    float x_squared[8][8];
    float mean_sq_sum[8][1];
    float mean_sq[8][1];
    float mean_sq_eps[8][1];
    float rms[8][1];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (2 ops): x=TLOAD(input,0,0); x_squared=TMUL(x,x)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v1, _v2);
            vst1q_f32(&x_squared[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
        }
    }

    // TROWSUM: mean_sq_sum = rowsum(x_squared)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x_squared[_row][_col];
        }
        mean_sq_sum[_row][0] = _sum;}

    // FUSED LOOP (3 ops): mean_sq=TDIVS(mean_sq_sum,8.0f); mean_sq_eps=TADDS(mean_sq,1e-05f); rms=TSQRT(mean_sq_eps)
    float32x4_t _vs4 = vdupq_n_f32(8.0f);
    float32x4_t _vs5 = vdupq_n_f32(1e-05f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v6 = vld1q_f32(&mean_sq_sum[_row][_col]);
            float32x4_t _vr7 = vdivq_f32(_v6, _vs4);
            vst1q_f32(&mean_sq[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&mean_sq[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v8, _vs5);
            vst1q_f32(&mean_sq_eps[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&mean_sq_eps[_row][_col]);
            float32x4_t _vr11 = vsqrtq_f32(_v10);
            vst1q_f32(&rms[_row][_col], _vr11);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            mean_sq[_row][_col] = mean_sq_sum[_row][_col] / 8.0f;
            mean_sq_eps[_row][_col] = mean_sq[_row][_col] + 1e-05f;
            rms[_row][_col] = sqrtf(mean_sq_eps[_row][_col]);
        }
    }

    // FUSED LOOP (2 ops): result=TDIV(x,rms); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&rms[_row][_col]);
            float32x4_t _vr14 = vdivq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] / rms[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_RMSNorm"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    nn_RMSNorm((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER