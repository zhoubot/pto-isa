// PTO Program: soft_capping_attention
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: soft_capping_attention
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     17
//   Total capacity (no reuse): 4,128 bytes (4.0 KB)
//   Total capacity (w/ reuse): 1,568 bytes (1.5 KB)
//   Reuse savings:            2,560 bytes (62.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  1,  -1]           -
//   Q                    8x8        f32       256   [  0,  -1]           -
//   V                    8x8        f32       256   [  2,  -1]           -
//   attn                 8x8        f32       256   [ 17,  -1]           <- capped_scores
//   capped_scores        8x8        f32       256   [ 11,  14]           <- exp_minus_1
//   exp_2x               8x8        f32       256   [  7,   9]           <- x_div_cap
//   exp_minus_1          8x8        f32       256   [  8,  10]           <- two_x
//   exp_plus_1           8x8        f32       256   [  9,  10]           -
//   exp_scores           8x8        f32       256   [ 15,  17]           <- tanh_x
//   output               8x8        f32       256   [ 18,  19]           <- shifted
//   row_sum              8x1        f32        32   [ 12,  17]           -
//   scaled               8x8        f32       256   [  4,   5]           -
//   scores               8x8        f32       256   [  3,   4]           -
//   shifted              8x8        f32       256   [ 14,  15]           <- exp_plus_1
//   tanh_x               8x8        f32       256   [ 10,  11]           <- exp_2x
//   two_x                8x8        f32       256   [  6,   7]           <- scaled
//   x_div_cap            8x8        f32       256   [  5,   6]           <- scores
//
// BUFFER REUSE MAP:
//   x_div_cap reuses buffer of scores
//   two_x reuses buffer of scaled
//   exp_2x reuses buffer of x_div_cap
//   exp_minus_1 reuses buffer of two_x
//   tanh_x reuses buffer of exp_2x
//   capped_scores reuses buffer of exp_minus_1
//   shifted reuses buffer of exp_plus_1
//   exp_scores reuses buffer of tanh_x
//   attn reuses buffer of capped_scores
//   output reuses buffer of shifted
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void soft_capping_attention(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    float Q[8][8];
    float K[8][8];
    float V[8][8];
    float scores[8][8];
    float scaled[8][8];
    float x_div_cap[8][8];
    float two_x[8][8];
    float exp_2x[8][8];
    float exp_minus_1[8][8];
    float exp_plus_1[8][8];
    float tanh_x[8][8];
    float capped_scores[8][8];
    float row_sum[8][1];
    float shifted[8][8];
    float exp_scores[8][8];
    float attn[8][8];
    float output[8][8];

    // Loop fusion: 10 loop overheads saved

    // FUSED LOOP (3 ops): Q=TLOAD(Q_mem,0,0); K=TLOAD(K_mem,0,0); V=TLOAD(V_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&Q_mem[_row * 8 + _col]);
            vst1q_f32(&Q[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&K_mem[_row * 8 + _col]);
            vst1q_f32(&K[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&V_mem[_row * 8 + _col]);
            vst1q_f32(&V[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            Q[_row][_col] = Q_mem[_row * 8 + _col];
            K[_row][_col] = K_mem[_row * 8 + _col];
            V[_row][_col] = V_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: scores = Q @ K
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += Q[_i][_k] * K[_k][_j];}
            scores[_i][_j] = _sum;}}

    // FUSED LOOP (8 ops): scaled=TMULS(scores,0.35355339059327373f); x_div_cap=TDIVS(scaled,50.0f); two_x=TMULS(x_div_cap,2.0f); exp_2x=TEXP(two_x); exp_minus_1=TADDS(exp_2x,-1.0f); exp_plus_1=TADDS(exp_2x,1.0f); tanh_x=TDIV(exp_minus_1,exp_plus_1); capped_scores=TMULS(tanh_x,50.0f)
    float32x4_t _vs3 = vdupq_n_f32(0.35355339059327373f);
    float32x4_t _vs4 = vdupq_n_f32(50.0f);
    float32x4_t _vs5 = vdupq_n_f32(2.0f);
    float32x4_t _vs6 = vdupq_n_f32(-1.0f);
    float32x4_t _vs7 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v8, _vs3);
            vst1q_f32(&scaled[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr11 = vdivq_f32(_v10, _vs4);
            vst1q_f32(&x_div_cap[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&x_div_cap[_row][_col]);
            float32x4_t _vr13 = vmulq_f32(_v12, _vs5);
            vst1q_f32(&two_x[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&two_x[_row][_col]);
            float32x4_t _vr15 = _v14;
            vst1q_f32(&exp_2x[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr17 = vaddq_f32(_v16, _vs6);
            vst1q_f32(&exp_minus_1[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr19 = vaddq_f32(_v18, _vs7);
            vst1q_f32(&exp_plus_1[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&exp_minus_1[_row][_col]);
            float32x4_t _v21 = vld1q_f32(&exp_plus_1[_row][_col]);
            float32x4_t _vr22 = vdivq_f32(_v20, _v21);
            vst1q_f32(&tanh_x[_row][_col], _vr22);
            float32x4_t _v23 = vld1q_f32(&tanh_x[_row][_col]);
            float32x4_t _vr24 = vmulq_f32(_v23, _vs4);
            vst1q_f32(&capped_scores[_row][_col], _vr24);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
            x_div_cap[_row][_col] = scaled[_row][_col] / 50.0f;
            two_x[_row][_col] = x_div_cap[_row][_col] * 2.0f;
            exp_2x[_row][_col] = expf(two_x[_row][_col]);
            exp_minus_1[_row][_col] = exp_2x[_row][_col] + -1.0f;
            exp_plus_1[_row][_col] = exp_2x[_row][_col] + 1.0f;
            tanh_x[_row][_col] = exp_minus_1[_row][_col] / exp_plus_1[_row][_col];
            capped_scores[_row][_col] = tanh_x[_row][_col] * 50.0f;
        }
    }

    // TROWSUM: row_sum = rowsum(capped_scores)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += capped_scores[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_sum=TDIVS(row_sum,8.0f)
    float32x4_t _vs25 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v26 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr27 = vdivq_f32(_v26, _vs25);
            vst1q_f32(&row_sum[_row][_col], _vr27);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // FUSED LOOP (2 ops): shifted=TROWEXPANDSUB(capped_scores,row_sum); exp_scores=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v028 = vld1q_f32(&capped_scores[_row][_col]);
            float32x4_t _vb30 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr29 = vsubq_f32(_v028, _vb30);
            vst1q_f32(&shifted[_row][_col], _vr29);
            float32x4_t _v31 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr32 = _v31;
            vst1q_f32(&exp_scores[_row][_col], _vr32);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            shifted[_row][_col] = capped_scores[_row][_col] - row_sum[_row][0];
            exp_scores[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_scores)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_scores[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): attn=TROWEXPANDDIV(exp_scores,row_sum)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v033 = vld1q_f32(&exp_scores[_row][_col]);
            float32x4_t _vb35 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr34 = vdivq_f32(_v033, _vb35);
            vst1q_f32(&attn[_row][_col], _vr34);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            attn[_row][_col] = exp_scores[_row][_col] / row_sum[_row][0];
        }
    }

    // TMATMUL: output = attn @ V
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += attn[_i][_k] * V[_k][_j];}
            output[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output_mem=TSTORE(output,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs36 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 8 + _col], _vs36);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output_mem[_row * 8 + _col] = output[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "soft_capping_attention"; }
enum { kPtoNumMemrefs = 4 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "Q_mem",
    "K_mem",
    "V_mem",
    "output_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    soft_capping_attention((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3]);
}
#endif  // PTO_CPU_SMOKE_RUNNER