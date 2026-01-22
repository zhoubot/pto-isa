// PTO Program: aten_silu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_silu
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
//   exp_neg              1x4096     f32     16384   [  5,  16]           -
//   neg_x                1x4096     f32     16384   [  4,  15]           -
//   one_plus             1x4096     f32     16384   [  6,  17]           -
//   result               1x4096     f32     16384   [  8,  19]           -
//   sigmoid_out          1x4096     f32     16384   [  7,  18]           -
//   x                    1x4096     f32     16384   [  3,  18]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void aten_silu(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    float x[1][4096];
    float neg_x[1][4096];
    float exp_neg[1][4096];
    float one_plus[1][4096];
    float sigmoid_out[1][4096];
    float result[1][4096];

    // Loop fusion: 12 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (7 ops): x=TLOAD(input,tile_idx,0); neg_x=TNEG(x); exp_neg=TEXP(neg_x); one_plus=TADDS(exp_neg,1.0f); sigmoid_out=TRECIP(one_plus); result=TMUL(x,sigmoid_out); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(1.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl1 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl1);
                float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr3 = vnegq_f32(_v2);
                vst1q_f32(&neg_x[_row][_col], _vr3);
                float32x4_t _v4 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr5 = _v4;
                vst1q_f32(&exp_neg[_row][_col], _vr5);
                float32x4_t _v6 = vld1q_f32(&exp_neg[_row][_col]);
                float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
                vst1q_f32(&one_plus[_row][_col], _vr7);
                float32x4_t _v8 = vld1q_f32(&one_plus[_row][_col]);
                float32x4_t _vr9 = _v8;
                vst1q_f32(&sigmoid_out[_row][_col], _vr9);
                float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v11 = vld1q_f32(&sigmoid_out[_row][_col]);
                float32x4_t _vr12 = vmulq_f32(_v10, _v11);
                vst1q_f32(&result[_row][_col], _vr12);
                float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs13);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg[_row][_col] = expf(neg_x[_row][_col]);
                one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
                sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
                result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (7 ops): x=TLOAD(input,num_full_tiles,0); neg_x=TNEG(x); exp_neg=TEXP(neg_x); one_plus=TADDS(exp_neg,1.0f); sigmoid_out=TRECIP(one_plus); result=TMUL(x,sigmoid_out); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs14 = vdupq_n_f32(1.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl15 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl15);
                float32x4_t _v16 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr17 = vnegq_f32(_v16);
                vst1q_f32(&neg_x[_row][_col], _vr17);
                float32x4_t _v18 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr19 = _v18;
                vst1q_f32(&exp_neg[_row][_col], _vr19);
                float32x4_t _v20 = vld1q_f32(&exp_neg[_row][_col]);
                float32x4_t _vr21 = vaddq_f32(_v20, _vs14);
                vst1q_f32(&one_plus[_row][_col], _vr21);
                float32x4_t _v22 = vld1q_f32(&one_plus[_row][_col]);
                float32x4_t _vr23 = _v22;
                vst1q_f32(&sigmoid_out[_row][_col], _vr23);
                float32x4_t _v24 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v25 = vld1q_f32(&sigmoid_out[_row][_col]);
                float32x4_t _vr26 = vmulq_f32(_v24, _v25);
                vst1q_f32(&result[_row][_col], _vr26);
                float32x4_t _vs27 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs27);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg[_row][_col] = expf(neg_x[_row][_col]);
                one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
                sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
                result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "aten_silu"; }
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
    aten_silu((float*)args[0], (float*)args[1], (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER