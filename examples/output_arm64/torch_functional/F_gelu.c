// PTO Program: F_gelu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_gelu
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     14
//   Total capacity (no reuse): 3,584 bytes (3.5 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            2,304 bytes (64.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   coeff_x3             8x8        f32       256   [  3,   4]           <- x_sq
//   cosh_approx          8x8        f32       256   [  9,  10]           -
//   exp_neg              8x8        f32       256   [-, -]               -
//   exp_pos              8x8        f32       256   [  7,   9]           <- inner
//   half_x               8x8        f32       256   [ 12,  13]           <- cosh_approx
//   inner                8x8        f32       256   [  4,   5]           <- x_cubed
//   one_plus             8x8        f32       256   [ 11,  13]           <- sinh_approx
//   result               8x8        f32       256   [ 13,  14]           <- x
//   scaled               8x8        f32       256   [  5,   7]           <- coeff_x3
//   sinh_approx          8x8        f32       256   [  8,  10]           <- scaled
//   tanh_out             8x8        f32       256   [ 10,  11]           <- exp_pos
//   x                    8x8        f32       256   [  0,  12]           -
//   x_cubed              8x8        f32       256   [  2,   3]           -
//   x_sq                 8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   coeff_x3 reuses buffer of x_sq
//   inner reuses buffer of x_cubed
//   scaled reuses buffer of coeff_x3
//   tanh_out reuses buffer of exp_pos
//   exp_pos reuses buffer of inner
//   sinh_approx reuses buffer of scaled
//   one_plus reuses buffer of sinh_approx
//   half_x reuses buffer of cosh_approx
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_gelu(float* input, float* output) {
    float x[8][8];
    float x_cubed[8][8];
    float x_sq[8][8];
    float coeff_x3[8][8];
    float inner[8][8];
    float scaled[8][8];
    float tanh_out[8][8];
    float exp_pos[8][8];
    float exp_neg[8][8];
    float sinh_approx[8][8];
    float cosh_approx[8][8];
    float one_plus[8][8];
    float half_x[8][8];
    float result[8][8];

    // Loop fusion: 14 loop overheads saved

    // FUSED LOOP (15 ops): x=TLOAD(input,0,0); x_sq=TMUL(x,x); x_cubed=TMUL(x_sq,x); coeff_x3=TMULS(x_cubed,0.044715f); inner=TADD(x,coeff_x3); scaled=TMULS(inner,0.7978845608028654f); scaled=TMULS(scaled,2.0f); exp_pos=TEXP(scaled); sinh_approx=TADDS(exp_pos,-1.0f); cosh_approx=TADDS(exp_pos,1.0f); tanh_out=TDIV(sinh_approx,cosh_approx); one_plus=TADDS(tanh_out,1.0f); half_x=TMULS(x,0.5f); result=TMUL(half_x,one_plus); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.044715f);
    float32x4_t _vs1 = vdupq_n_f32(0.7978845608028654f);
    float32x4_t _vs2 = vdupq_n_f32(2.0f);
    float32x4_t _vs3 = vdupq_n_f32(-1.0f);
    float32x4_t _vs4 = vdupq_n_f32(1.0f);
    float32x4_t _vs5 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl6 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl6);
            float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v7, _v8);
            vst1q_f32(&x_sq[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x_sq[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&x_cubed[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&x_cubed[_row][_col]);
            float32x4_t _vr14 = vmulq_f32(_v13, _vs0);
            vst1q_f32(&coeff_x3[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v16 = vld1q_f32(&coeff_x3[_row][_col]);
            float32x4_t _vr17 = vaddq_f32(_v15, _v16);
            vst1q_f32(&inner[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&inner[_row][_col]);
            float32x4_t _vr19 = vmulq_f32(_v18, _vs1);
            vst1q_f32(&scaled[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr21 = vmulq_f32(_v20, _vs2);
            vst1q_f32(&scaled[_row][_col], _vr21);
            float32x4_t _v22 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr23 = _v22;
            vst1q_f32(&exp_pos[_row][_col], _vr23);
            float32x4_t _v24 = vld1q_f32(&exp_pos[_row][_col]);
            float32x4_t _vr25 = vaddq_f32(_v24, _vs3);
            vst1q_f32(&sinh_approx[_row][_col], _vr25);
            float32x4_t _v26 = vld1q_f32(&exp_pos[_row][_col]);
            float32x4_t _vr27 = vaddq_f32(_v26, _vs4);
            vst1q_f32(&cosh_approx[_row][_col], _vr27);
            float32x4_t _v28 = vld1q_f32(&sinh_approx[_row][_col]);
            float32x4_t _v29 = vld1q_f32(&cosh_approx[_row][_col]);
            float32x4_t _vr30 = vdivq_f32(_v28, _v29);
            vst1q_f32(&tanh_out[_row][_col], _vr30);
            float32x4_t _v31 = vld1q_f32(&tanh_out[_row][_col]);
            float32x4_t _vr32 = vaddq_f32(_v31, _vs4);
            vst1q_f32(&one_plus[_row][_col], _vr32);
            float32x4_t _v33 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr34 = vmulq_f32(_v33, _vs5);
            vst1q_f32(&half_x[_row][_col], _vr34);
            float32x4_t _v35 = vld1q_f32(&half_x[_row][_col]);
            float32x4_t _v36 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr37 = vmulq_f32(_v35, _v36);
            vst1q_f32(&result[_row][_col], _vr37);
            float32x4_t _vs38 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs38);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
            x_cubed[_row][_col] = x_sq[_row][_col] * x[_row][_col];
            coeff_x3[_row][_col] = x_cubed[_row][_col] * 0.044715f;
            inner[_row][_col] = x[_row][_col] + coeff_x3[_row][_col];
            scaled[_row][_col] = inner[_row][_col] * 0.7978845608028654f;
            scaled[_row][_col] = scaled[_row][_col] * 2.0f;
            exp_pos[_row][_col] = expf(scaled[_row][_col]);
            sinh_approx[_row][_col] = exp_pos[_row][_col] + -1.0f;
            cosh_approx[_row][_col] = exp_pos[_row][_col] + 1.0f;
            tanh_out[_row][_col] = sinh_approx[_row][_col] / cosh_approx[_row][_col];
            one_plus[_row][_col] = tanh_out[_row][_col] + 1.0f;
            half_x[_row][_col] = x[_row][_col] * 0.5f;
            result[_row][_col] = half_x[_row][_col] * one_plus[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_gelu"; }
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
    F_gelu((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER