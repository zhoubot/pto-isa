// PTO Program: aten_cosh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_cosh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_x            1x4096     f32     16384   [  6,  17]           -
//   exp_x                1x4096     f32     16384   [  4,  17]           -
//   neg_x                1x4096     f32     16384   [  5,  16]           -
//   result               1x4096     f32     16384   [  8,  19]           -
//   sum                  1x4096     f32     16384   [  7,  18]           -
//   x                    1x4096     f32     16384   [  3,  15]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void aten_cosh(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    float x[1][4096];
    float neg_x[1][4096];
    float exp_x[1][4096];
    float exp_neg_x[1][4096];
    float sum[1][4096];
    float result[1][4096];

    // Loop fusion: 12 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (7 ops): x=TLOAD(input,tile_idx,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(2.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl1 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl1);
                float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr3 = _v2;
                vst1q_f32(&exp_x[_row][_col], _vr3);
                float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr5 = vnegq_f32(_v4);
                vst1q_f32(&neg_x[_row][_col], _vr5);
                float32x4_t _v6 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr7 = _v6;
                vst1q_f32(&exp_neg_x[_row][_col], _vr7);
                float32x4_t _v8 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v9 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr10 = vaddq_f32(_v8, _v9);
                vst1q_f32(&sum[_row][_col], _vr10);
                float32x4_t _v11 = vld1q_f32(&sum[_row][_col]);
                float32x4_t _vr12 = vdivq_f32(_v11, _vs0);
                vst1q_f32(&result[_row][_col], _vr12);
                float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs13);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = sum[_row][_col] / 2.0f;
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (7 ops): x=TLOAD(input,num_full_tiles,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs14 = vdupq_n_f32(2.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl15 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl15);
                float32x4_t _v16 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr17 = _v16;
                vst1q_f32(&exp_x[_row][_col], _vr17);
                float32x4_t _v18 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr19 = vnegq_f32(_v18);
                vst1q_f32(&neg_x[_row][_col], _vr19);
                float32x4_t _v20 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr21 = _v20;
                vst1q_f32(&exp_neg_x[_row][_col], _vr21);
                float32x4_t _v22 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v23 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr24 = vaddq_f32(_v22, _v23);
                vst1q_f32(&sum[_row][_col], _vr24);
                float32x4_t _v25 = vld1q_f32(&sum[_row][_col]);
                float32x4_t _vr26 = vdivq_f32(_v25, _vs14);
                vst1q_f32(&result[_row][_col], _vr26);
                float32x4_t _vs27 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs27);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = sum[_row][_col] / 2.0f;
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "aten_cosh"; }
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
    aten_cosh((float*)args[0], (float*)args[1], (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER