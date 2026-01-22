// PTO Program: tensor_cos
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_cos
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 2,048 bytes (2.0 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            1,024 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   ones                 8x8        f32       256   [  5,   6]           <- x4
//   result               8x8        f32       256   [  7,   8]           <- term1
//   temp                 8x8        f32       256   [  6,   7]           -
//   term1                8x8        f32       256   [  3,   6]           -
//   term2                8x8        f32       256   [  4,   7]           <- x2
//   x                    8x8        f32       256   [  0,   1]           -
//   x2                   8x8        f32       256   [  1,   3]           -
//   x4                   8x8        f32       256   [  2,   4]           <- x
//
// BUFFER REUSE MAP:
//   x4 reuses buffer of x
//   term2 reuses buffer of x2
//   ones reuses buffer of x4
//   result reuses buffer of term1
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_cos(float* input, float* output) {
    float x[8][8];
    float x2[8][8];
    float x4[8][8];
    float term1[8][8];
    float term2[8][8];
    float ones[8][8];
    float temp[8][8];
    float result[8][8];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (9 ops): x=TLOAD(input,0,0); x2=TMUL(x,x); x4=TMUL(x2,x2); term1=TDIVS(x2,2.0f); term2=TDIVS(x4,24.0f); ones=TEXPANDS(1.0f); temp=TSUB(ones,term1); result=TADD(temp,term2); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    float32x4_t _vs1 = vdupq_n_f32(24.0f);
    float32x4_t _vs2 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&x2[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v7, _v8);
            vst1q_f32(&x4[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr11 = vdivq_f32(_v10, _vs0);
            vst1q_f32(&term1[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&x4[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs1);
            vst1q_f32(&term2[_row][_col], _vr13);
            vst1q_f32(&ones[_row][_col], _vs2);
            float32x4_t _v14 = vld1q_f32(&ones[_row][_col]);
            float32x4_t _v15 = vld1q_f32(&term1[_row][_col]);
            float32x4_t _vr16 = vsubq_f32(_v14, _v15);
            vst1q_f32(&temp[_row][_col], _vr16);
            float32x4_t _v17 = vld1q_f32(&temp[_row][_col]);
            float32x4_t _v18 = vld1q_f32(&term2[_row][_col]);
            float32x4_t _vr19 = vaddq_f32(_v17, _v18);
            vst1q_f32(&result[_row][_col], _vr19);
            float32x4_t _vs20 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs20);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x2[_row][_col] = x[_row][_col] * x[_row][_col];
            x4[_row][_col] = x2[_row][_col] * x2[_row][_col];
            term1[_row][_col] = x2[_row][_col] / 2.0f;
            term2[_row][_col] = x4[_row][_col] / 24.0f;
            ones[_row][_col] = 1.0f;
            temp[_row][_col] = ones[_row][_col] - term1[_row][_col];
            result[_row][_col] = temp[_row][_col] + term2[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_cos"; }
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
    tensor_cos((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER