// PTO Program: nn_Hardsigmoid
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Hardsigmoid
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
//   relu6_out            8x8        f32       256   [  4,   5]           -
//   relu_out             8x8        f32       256   [  2,   4]           <- x
//   result               8x8        f32       256   [  5,   6]           <- relu_out
//   six                  8x8        f32       256   [  3,   4]           <- x_plus_3
//   x                    8x8        f32       256   [  0,   1]           -
//   x_plus_3             8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   relu_out reuses buffer of x
//   six reuses buffer of x_plus_3
//   result reuses buffer of relu_out
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_Hardsigmoid(float* input, float* output) {
    float x[8][8];
    float x_plus_3[8][8];
    float relu_out[8][8];
    float six[8][8];
    float relu6_out[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); x_plus_3=TADDS(x,3.0f); relu_out=TRELU(x_plus_3); six=TEXPANDS(6.0f); relu6_out=TMIN(relu_out,six); result=TDIVS(relu6_out,6.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(3.0f);
    float32x4_t _vs1 = vdupq_n_f32(6.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vaddq_f32(_v3, _vs0);
            vst1q_f32(&x_plus_3[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&x_plus_3[_row][_col]);
            float32x4_t _vr6 = vmaxq_f32(_v5, vdupq_n_f32(0.0f));
            vst1q_f32(&relu_out[_row][_col], _vr6);
            vst1q_f32(&six[_row][_col], _vs1);
            float32x4_t _v7 = vld1q_f32(&relu_out[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&six[_row][_col]);
            float32x4_t _vr9 = vminq_f32(_v7, _v8);
            vst1q_f32(&relu6_out[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&relu6_out[_row][_col]);
            float32x4_t _vr11 = vdivq_f32(_v10, _vs1);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
            relu_out[_row][_col] = fmaxf(x_plus_3[_row][_col], 0.0f);
            six[_row][_col] = 6.0f;
            relu6_out[_row][_col] = relu_out[_row][_col] + six[_row][_col];
            result[_row][_col] = relu6_out[_row][_col] / 6.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_Hardsigmoid"; }
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
    nn_Hardsigmoid((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER