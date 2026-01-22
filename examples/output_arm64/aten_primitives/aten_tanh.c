// PTO Program: aten_tanh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_tanh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 114,688 bytes (112.0 KB)
//   Total capacity (w/ reuse): 114,688 bytes (112.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   denominator          1x4096     f32     16384   [  8,  20]           -
//   exp_neg_x            1x4096     f32     16384   [  6,  19]           -
//   exp_x                1x4096     f32     16384   [  4,  19]           -
//   neg_x                1x4096     f32     16384   [  5,  17]           -
//   numerator            1x4096     f32     16384   [  7,  20]           -
//   result               1x4096     f32     16384   [  9,  21]           -
//   x                    1x4096     f32     16384   [  3,  16]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void aten_tanh(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    float x[1][4096];
    float exp_x[1][4096];
    float exp_neg_x[1][4096];
    float neg_x[1][4096];
    float numerator[1][4096];
    float denominator[1][4096];
    float result[1][4096];

    // Loop fusion: 14 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (8 ops): x=TLOAD(input,tile_idx,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); numerator=TSUB(exp_x,exp_neg_x); denominator=TADD(exp_x,exp_neg_x); result=TDIV(numerator,denominator); output=TSTORE(result,tile_idx,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl0 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl0);
                float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr2 = _v1;
                vst1q_f32(&exp_x[_row][_col], _vr2);
                float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr4 = vnegq_f32(_v3);
                vst1q_f32(&neg_x[_row][_col], _vr4);
                float32x4_t _v5 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr6 = _v5;
                vst1q_f32(&exp_neg_x[_row][_col], _vr6);
                float32x4_t _v7 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v8 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr9 = vsubq_f32(_v7, _v8);
                vst1q_f32(&numerator[_row][_col], _vr9);
                float32x4_t _v10 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v11 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr12 = vaddq_f32(_v10, _v11);
                vst1q_f32(&denominator[_row][_col], _vr12);
                float32x4_t _v13 = vld1q_f32(&numerator[_row][_col]);
                float32x4_t _v14 = vld1q_f32(&denominator[_row][_col]);
                float32x4_t _vr15 = vdivq_f32(_v13, _v14);
                vst1q_f32(&result[_row][_col], _vr15);
                float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs16);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
                denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (8 ops): x=TLOAD(input,num_full_tiles,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); numerator=TSUB(exp_x,exp_neg_x); denominator=TADD(exp_x,exp_neg_x); result=TDIV(numerator,denominator); output=TSTORE(result,num_full_tiles,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl17 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl17);
                float32x4_t _v18 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr19 = _v18;
                vst1q_f32(&exp_x[_row][_col], _vr19);
                float32x4_t _v20 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr21 = vnegq_f32(_v20);
                vst1q_f32(&neg_x[_row][_col], _vr21);
                float32x4_t _v22 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr23 = _v22;
                vst1q_f32(&exp_neg_x[_row][_col], _vr23);
                float32x4_t _v24 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v25 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr26 = vsubq_f32(_v24, _v25);
                vst1q_f32(&numerator[_row][_col], _vr26);
                float32x4_t _v27 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v28 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr29 = vaddq_f32(_v27, _v28);
                vst1q_f32(&denominator[_row][_col], _vr29);
                float32x4_t _v30 = vld1q_f32(&numerator[_row][_col]);
                float32x4_t _v31 = vld1q_f32(&denominator[_row][_col]);
                float32x4_t _vr32 = vdivq_f32(_v30, _v31);
                vst1q_f32(&result[_row][_col], _vr32);
                float32x4_t _vs33 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs33);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
                denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "aten_tanh"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
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
    aten_tanh((float*)args[0], (float*)args[1], (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER