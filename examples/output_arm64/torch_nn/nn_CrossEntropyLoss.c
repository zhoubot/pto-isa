// PTO Program: nn_CrossEntropyLoss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_CrossEntropyLoss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 1,640 bytes (1.6 KB)
//   Total capacity (w/ reuse): 840 bytes (0.8 KB)
//   Reuse savings:            800 bytes (48.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_pred             8x8        f32       256   [  2,   3]           -
//   log_softmax          8x8        f32       256   [  5,   6]           <- exp_pred
//   log_sum              8x1        f32        32   [  4,   5]           -
//   neg_weighted         8x8        f32       256   [  7,   8]           <- target
//   pred                 8x8        f32       256   [  0,   5]           -
//   result               1x1        f32         4   [ 10,  11]           -
//   row_sum              8x1        f32        32   [  8,   9]           <- sum_exp
//   sum_exp              8x1        f32        32   [  3,   4]           -
//   target               8x8        f32       256   [  1,   6]           -
//   total_sum            1x1        f32         4   [  9,  10]           -
//   weighted             8x8        f32       256   [  6,   7]           <- pred
//
// BUFFER REUSE MAP:
//   log_softmax reuses buffer of exp_pred
//   weighted reuses buffer of pred
//   neg_weighted reuses buffer of target
//   row_sum reuses buffer of sum_exp
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_CrossEntropyLoss(float* pred_mem, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float exp_pred[8][8];
    float sum_exp[8][1];
    float log_sum[8][1];
    float log_softmax[8][8];
    float weighted[8][8];
    float neg_weighted[8][8];
    float row_sum[8][1];
    float total_sum[1][1];
    float result[1][1];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (3 ops): pred=TLOAD(pred_mem,0,0); target=TLOAD(target_mem,0,0); exp_pred=TEXP(pred)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&pred_mem[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&exp_pred[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = pred_mem[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            exp_pred[_row][_col] = expf(pred[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_pred)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_pred[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(sum_exp)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&sum_exp[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&log_sum[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(sum_exp[_row][_col]);
        }
    }

    // FUSED LOOP (3 ops): log_softmax=TROWEXPANDSUB(pred,log_sum); weighted=TMUL(target,log_softmax); neg_weighted=TNEG(weighted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v06 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _vb8 = vdupq_n_f32(log_sum[_row][0]);
            float32x4_t _vr7 = vsubq_f32(_v06, _vb8);
            vst1q_f32(&log_softmax[_row][_col], _vr7);
            float32x4_t _v9 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&log_softmax[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&weighted[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&weighted[_row][_col]);
            float32x4_t _vr13 = vnegq_f32(_v12);
            vst1q_f32(&neg_weighted[_row][_col], _vr13);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            log_softmax[_row][_col] = pred[_row][_col] - log_sum[_row][0];
            weighted[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
            neg_weighted[_row][_col] = -weighted[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(neg_weighted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += neg_weighted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total_sum = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total_sum[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(total_sum,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs14 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v15 = vld1q_f32(&total_sum[_row][_col]);
            float32x4_t _vr16 = vdivq_f32(_v15, _vs14);
            vst1q_f32(&result[_row][_col], _vr16);
            float32x4_t _vs17 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs17);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = total_sum[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_CrossEntropyLoss"; }
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
    nn_CrossEntropyLoss((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER