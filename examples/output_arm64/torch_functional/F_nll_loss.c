// PTO Program: F_nll_loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_nll_loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 804 bytes (0.8 KB)
//   Total capacity (w/ reuse): 804 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_probs            8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [  4,   7]           -
//   row_sum              8x1        f32        32   [  3,   4]           -
//   target               8x8        f32       256   [  1,   2]           -
//   weighted             8x8        f32       256   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_nll_loss(float* input, float* target_mem, float* output) {
    float log_probs[8][8];
    float target[8][8];
    float weighted[8][8];
    float row_sum[8][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (3 ops): log_probs=TLOAD(input,0,0); target=TLOAD(target_mem,0,0); weighted=TMUL(target,log_probs)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&log_probs[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&log_probs[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&weighted[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            log_probs[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            weighted[_row][_col] = target[_row][_col] * log_probs[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(weighted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += weighted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: result = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        result[0][_col] = _sum;}

    // FUSED LOOP (3 ops): result=TNEG(result); result=TDIVS(result,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs5 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v6 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr7 = vnegq_f32(_v6);
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr9 = vdivq_f32(_v8, _vs5);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = -result[_row][_col];
            result[_row][_col] = result[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_nll_loss"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
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
    F_nll_loss((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER