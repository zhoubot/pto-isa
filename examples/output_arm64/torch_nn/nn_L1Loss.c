// PTO Program: nn_L1Loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_L1Loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,064 bytes (1.0 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            256 bytes (24.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_diff             8x8        f32       256   [  3,   4]           <- pred
//   diff                 8x8        f32       256   [  2,   3]           -
//   pred                 8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [  6,   7]           -
//   row_sum              8x1        f32        32   [  4,   5]           -
//   target               8x8        f32       256   [  1,   2]           -
//   total_sum            1x1        f32         4   [  5,   6]           -
//
// BUFFER REUSE MAP:
//   abs_diff reuses buffer of pred
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_L1Loss(float* pred_mem, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float diff[8][8];
    float abs_diff[8][8];
    float row_sum[8][1];
    float total_sum[1][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (4 ops): pred=TLOAD(pred_mem,0,0); target=TLOAD(target_mem,0,0); diff=TSUB(pred,target); abs_diff=TABS(diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&pred_mem[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr4 = vsubq_f32(_v2, _v3);
            vst1q_f32(&diff[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr6 = vabsq_f32(_v5);
            vst1q_f32(&abs_diff[_row][_col], _vr6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = pred_mem[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            diff[_row][_col] = pred[_row][_col] - target[_row][_col];
            abs_diff[_row][_col] = fabsf(diff[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(abs_diff)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += abs_diff[_row][_col];
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
    float32x4_t _vs7 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&total_sum[_row][_col]);
            float32x4_t _vr9 = vdivq_f32(_v8, _vs7);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs10);
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
const char* pto_program_name() { return "nn_L1Loss"; }
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
    nn_L1Loss((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER