// PTO Program: nn_Mish
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Mish
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,816 bytes (2.8 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            1,536 bytes (54.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_sp           8x8        f32       256   [  6,   8]           <- softplus
//   exp_sp               8x8        f32       256   [  4,   8]           <- one_plus_exp
//   exp_x                8x8        f32       256   [  1,   2]           -
//   neg_sp               8x8        f32       256   [  5,   6]           -
//   one_plus_exp         8x8        f32       256   [  2,   3]           -
//   result               8x8        f32       256   [ 10,  11]           <- exp_neg_sp
//   softplus             8x8        f32       256   [  3,   5]           <- exp_x
//   tanh_den             8x8        f32       256   [  8,   9]           -
//   tanh_num             8x8        f32       256   [  7,   9]           <- neg_sp
//   tanh_out             8x8        f32       256   [  9,  10]           <- exp_sp
//   x                    8x8        f32       256   [  0,  10]           -
//
// BUFFER REUSE MAP:
//   softplus reuses buffer of exp_x
//   exp_sp reuses buffer of one_plus_exp
//   exp_neg_sp reuses buffer of softplus
//   tanh_num reuses buffer of neg_sp
//   tanh_out reuses buffer of exp_sp
//   result reuses buffer of exp_neg_sp
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_Mish(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float one_plus_exp[8][8];
    float softplus[8][8];
    float exp_sp[8][8];
    float neg_sp[8][8];
    float exp_neg_sp[8][8];
    float tanh_num[8][8];
    float tanh_den[8][8];
    float tanh_out[8][8];
    float result[8][8];

    // Loop fusion: 11 loop overheads saved

    // FUSED LOOP (12 ops): x=TLOAD(input,0,0); exp_x=TEXP(x); one_plus_exp=TADDS(exp_x,1.0f); softplus=TLOG(one_plus_exp); exp_sp=TEXP(softplus); neg_sp=TNEG(softplus); exp_neg_sp=TEXP(neg_sp); tanh_num=TSUB(exp_sp,exp_neg_sp); tanh_den=TADD(exp_sp,exp_neg_sp); tanh_out=TDIV(tanh_num,tanh_den); result=TMUL(x,tanh_out); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&exp_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v4, _vs0);
            vst1q_f32(&one_plus_exp[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&one_plus_exp[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&softplus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&softplus[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&exp_sp[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&softplus[_row][_col]);
            float32x4_t _vr11 = vnegq_f32(_v10);
            vst1q_f32(&neg_sp[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&neg_sp[_row][_col]);
            float32x4_t _vr13 = _v12;
            vst1q_f32(&exp_neg_sp[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&exp_sp[_row][_col]);
            float32x4_t _v15 = vld1q_f32(&exp_neg_sp[_row][_col]);
            float32x4_t _vr16 = vsubq_f32(_v14, _v15);
            vst1q_f32(&tanh_num[_row][_col], _vr16);
            float32x4_t _v17 = vld1q_f32(&exp_sp[_row][_col]);
            float32x4_t _v18 = vld1q_f32(&exp_neg_sp[_row][_col]);
            float32x4_t _vr19 = vaddq_f32(_v17, _v18);
            vst1q_f32(&tanh_den[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&tanh_num[_row][_col]);
            float32x4_t _v21 = vld1q_f32(&tanh_den[_row][_col]);
            float32x4_t _vr22 = vdivq_f32(_v20, _v21);
            vst1q_f32(&tanh_out[_row][_col], _vr22);
            float32x4_t _v23 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v24 = vld1q_f32(&tanh_out[_row][_col]);
            float32x4_t _vr25 = vmulq_f32(_v23, _v24);
            vst1q_f32(&result[_row][_col], _vr25);
            float32x4_t _vs26 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs26);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            one_plus_exp[_row][_col] = exp_x[_row][_col] + 1.0f;
            softplus[_row][_col] = logf(one_plus_exp[_row][_col]);
            exp_sp[_row][_col] = expf(softplus[_row][_col]);
            neg_sp[_row][_col] = -softplus[_row][_col];
            exp_neg_sp[_row][_col] = expf(neg_sp[_row][_col]);
            tanh_num[_row][_col] = exp_sp[_row][_col] - exp_neg_sp[_row][_col];
            tanh_den[_row][_col] = exp_sp[_row][_col] + exp_neg_sp[_row][_col];
            tanh_out[_row][_col] = tanh_num[_row][_col] / tanh_den[_row][_col];
            result[_row][_col] = x[_row][_col] * tanh_out[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_Mish"; }
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
    nn_Mish((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER