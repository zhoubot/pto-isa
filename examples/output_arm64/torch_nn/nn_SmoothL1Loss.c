// PTO Program: nn_SmoothL1Loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_SmoothL1Loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,088 bytes (2.0 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            1,280 bytes (61.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_diff             8x8        f32       256   [  3,   6]           <- pred
//   diff                 8x8        f32       256   [  2,   4]           -
//   l1_term              8x8        f32       256   [  6,   7]           <- squared
//   l2_term              8x8        f32       256   [  5,   7]           <- diff
//   pred                 8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [ 10,  11]           -
//   row_sum              8x1        f32        32   [  8,   9]           -
//   smooth               8x8        f32       256   [  7,   8]           <- abs_diff
//   squared              8x8        f32       256   [  4,   5]           <- target
//   target               8x8        f32       256   [  1,   2]           -
//   total_sum            1x1        f32         4   [  9,  10]           -
//
// BUFFER REUSE MAP:
//   abs_diff reuses buffer of pred
//   squared reuses buffer of target
//   l2_term reuses buffer of diff
//   l1_term reuses buffer of squared
//   smooth reuses buffer of abs_diff
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_SmoothL1Loss(float* pred_mem, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float diff[8][8];
    float abs_diff[8][8];
    float squared[8][8];
    float l2_term[8][8];
    float l1_term[8][8];
    float smooth[8][8];
    float row_sum[8][1];
    float total_sum[1][1];
    float result[1][1];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (8 ops): pred=TLOAD(pred_mem,0,0); target=TLOAD(target_mem,0,0); diff=TSUB(pred,target); abs_diff=TABS(diff); squared=TMUL(diff,diff); l2_term=TDIVS(squared,2.0f); l1_term=TADDS(abs_diff,-0.5f); smooth=TMIN(l2_term,l1_term)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    float32x4_t _vs1 = vdupq_n_f32(-0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&pred_mem[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl2);
            float32x4_t _vl3 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr6 = vsubq_f32(_v4, _v5);
            vst1q_f32(&diff[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr8 = vabsq_f32(_v7);
            vst1q_f32(&abs_diff[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&squared[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&squared[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs0);
            vst1q_f32(&l2_term[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&abs_diff[_row][_col]);
            float32x4_t _vr15 = vaddq_f32(_v14, _vs1);
            vst1q_f32(&l1_term[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&l2_term[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&l1_term[_row][_col]);
            float32x4_t _vr18 = vminq_f32(_v16, _v17);
            vst1q_f32(&smooth[_row][_col], _vr18);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = pred_mem[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            diff[_row][_col] = pred[_row][_col] - target[_row][_col];
            abs_diff[_row][_col] = fabsf(diff[_row][_col]);
            squared[_row][_col] = diff[_row][_col] * diff[_row][_col];
            l2_term[_row][_col] = squared[_row][_col] / 2.0f;
            l1_term[_row][_col] = abs_diff[_row][_col] + -0.5f;
            smooth[_row][_col] = l2_term[_row][_col] + l1_term[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(smooth)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += smooth[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total_sum = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total_sum[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(total_sum,64.0f); output=TSTORE(result,0,0)
    float32x4_t _vs19 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v20 = vld1q_f32(&total_sum[_row][_col]);
            float32x4_t _vr21 = vdivq_f32(_v20, _vs19);
            vst1q_f32(&result[_row][_col], _vr21);
            float32x4_t _vs22 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs22);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = total_sum[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_SmoothL1Loss"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "pred_mem",
    "target_mem",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(4),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
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
    nn_SmoothL1Loss((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER