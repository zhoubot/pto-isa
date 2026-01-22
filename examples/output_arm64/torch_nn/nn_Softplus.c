// PTO Program: nn_Softplus
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Softplus
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            1,024 bytes (66.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                8x8        f32       256   [  2,   3]           <- x
//   log_out              8x8        f32       256   [  4,   5]           <- exp_x
//   one_plus             8x8        f32       256   [  3,   4]           <- scaled_x
//   result               8x8        f32       256   [  5,   6]           <- one_plus
//   scaled_x             8x8        f32       256   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   one_plus reuses buffer of scaled_x
//   log_out reuses buffer of exp_x
//   result reuses buffer of one_plus
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_Softplus(float* input, float* output) {
    float x[8][8];
    float scaled_x[8][8];
    float exp_x[8][8];
    float one_plus[8][8];
    float log_out[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); scaled_x=TMULS(x,1.0f); exp_x=TEXP(scaled_x); one_plus=TADDS(exp_x,1.0f); log_out=TLOG(one_plus); result=TDIVS(log_out,1.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v2, _vs0);
            vst1q_f32(&scaled_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&scaled_x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&log_out[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&log_out[_row][_col]);
            float32x4_t _vr11 = vdivq_f32(_v10, _vs0);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.0f;
            exp_x[_row][_col] = expf(scaled_x[_row][_col]);
            one_plus[_row][_col] = exp_x[_row][_col] + 1.0f;
            log_out[_row][_col] = logf(one_plus[_row][_col]);
            result[_row][_col] = log_out[_row][_col] / 1.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_Softplus"; }
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
    nn_Softplus((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER