// PTO Program: attention_sliding_window
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_sliding_window
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 2,848 bytes (2.8 KB)
//   Total capacity (w/ reuse): 1,568 bytes (1.5 KB)
//   Reuse savings:            1,280 bytes (44.9%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  1,  -1]           -
//   Q                    8x8        f32       256   [  0,  -1]           -
//   V                    8x8        f32       256   [  2,  -1]           -
//   attn                 8x8        f32       256   [ 12,  -1]           <- masked_scores
//   exp_scores           8x8        f32       256   [ 10,  12]           <- scaled
//   masked_scores        8x8        f32       256   [  6,   9]           <- scores
//   output               8x8        f32       256   [ 13,  14]           <- shifted
//   row_sum              8x1        f32        32   [  7,  12]           -
//   scaled               8x8        f32       256   [  5,   6]           -
//   scores               8x8        f32       256   [  4,   5]           -
//   shifted              8x8        f32       256   [  9,  10]           <- window_mask
//   window_mask          8x8        f32       256   [  3,   6]           -
//
// BUFFER REUSE MAP:
//   masked_scores reuses buffer of scores
//   shifted reuses buffer of window_mask
//   exp_scores reuses buffer of scaled
//   attn reuses buffer of masked_scores
//   output reuses buffer of shifted
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_sliding_window(float* Q_mem, float* K_mem, float* V_mem, float* mask_mem, float* output_mem) {
    float Q[8][8];
    float K[8][8];
    float V[8][8];
    float scores[8][8];
    float scaled[8][8];
    float window_mask[8][8];
    float masked_scores[8][8];
    float row_sum[8][1];
    float shifted[8][8];
    float exp_scores[8][8];
    float attn[8][8];
    float output[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (4 ops): Q=TLOAD(Q_mem,0,0); K=TLOAD(K_mem,0,0); V=TLOAD(V_mem,0,0); window_mask=TLOAD(mask_mem,0,0)
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
            float32x4_t _vl3 = vld1q_f32(&mask_mem[_row * 8 + _col]);
            vst1q_f32(&window_mask[_row][_col], _vl3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            Q[_row][_col] = Q_mem[_row * 8 + _col];
            K[_row][_col] = K_mem[_row * 8 + _col];
            V[_row][_col] = V_mem[_row * 8 + _col];
            window_mask[_row][_col] = mask_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: scores = Q @ K
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += Q[_i][_k] * K[_k][_j];}
            scores[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): scaled=TMULS(scores,0.35355339059327373f); masked_scores=TADD(scaled,window_mask)
    float32x4_t _vs4 = vdupq_n_f32(0.35355339059327373f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v5 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v5, _vs4);
            vst1q_f32(&scaled[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&window_mask[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v7, _v8);
            vst1q_f32(&masked_scores[_row][_col], _vr9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
            masked_scores[_row][_col] = scaled[_row][_col] + window_mask[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(masked_scores)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += masked_scores[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_sum=TDIVS(row_sum,8.0f)
    float32x4_t _vs10 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v11 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr12 = vdivq_f32(_v11, _vs10);
            vst1q_f32(&row_sum[_row][_col], _vr12);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // FUSED LOOP (2 ops): shifted=TROWEXPANDSUB(masked_scores,row_sum); exp_scores=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v013 = vld1q_f32(&masked_scores[_row][_col]);
            float32x4_t _vb15 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr14 = vsubq_f32(_v013, _vb15);
            vst1q_f32(&shifted[_row][_col], _vr14);
            float32x4_t _v16 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr17 = _v16;
            vst1q_f32(&exp_scores[_row][_col], _vr17);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            shifted[_row][_col] = masked_scores[_row][_col] - row_sum[_row][0];
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
            float32x4_t _v018 = vld1q_f32(&exp_scores[_row][_col]);
            float32x4_t _vb20 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr19 = vdivq_f32(_v018, _vb20);
            vst1q_f32(&attn[_row][_col], _vr19);
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
            float32x4_t _vs21 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 8 + _col], _vs21);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output_mem[_row * 8 + _col] = output[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "attention_sliding_window"; }
enum { kPtoNumMemrefs = 5 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "Q_mem",
    "K_mem",
    "V_mem",
    "mask_mem",
    "output_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
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
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
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
    attention_sliding_window((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4]);
}
#endif  // PTO_CPU_SMOKE_RUNNER