// PTO Program: F_tanh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_tanh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            768 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   denominator          8x8        f32       256   [  4,   5]           -
//   exp_2x               8x8        f32       256   [  2,   4]           <- x
//   numerator            8x8        f32       256   [  3,   5]           <- x_2
//   result               8x8        f32       256   [  5,   6]           <- exp_2x
//   x                    8x8        f32       256   [  0,   1]           -
//   x_2                  8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   exp_2x reuses buffer of x
//   numerator reuses buffer of x_2
//   result reuses buffer of exp_2x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_tanh(float* input, float* output) {
    float x[8][8];
    float x_2[8][8];
    float exp_2x[8][8];
    float numerator[8][8];
    float denominator[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); x_2=TMULS(x,2.0f); exp_2x=TEXP(x_2); numerator=TADDS(exp_2x,-1.0f); denominator=TADDS(exp_2x,1.0f); result=TDIV(numerator,denominator); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    float32x4_t _vs1 = vdupq_n_f32(-1.0f);
    float32x4_t _vs2 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs0);
            vst1q_f32(&x_2[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x_2[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&exp_2x[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v8, _vs1);
            vst1q_f32(&numerator[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v10, _vs2);
            vst1q_f32(&denominator[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&numerator[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&denominator[_row][_col]);
            float32x4_t _vr14 = vdivq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_2[_row][_col] = x[_row][_col] * 2.0f;
            exp_2x[_row][_col] = expf(x_2[_row][_col]);
            numerator[_row][_col] = exp_2x[_row][_col] + -1.0f;
            denominator[_row][_col] = exp_2x[_row][_col] + 1.0f;
            result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_tanh"; }
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
    F_tanh((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER