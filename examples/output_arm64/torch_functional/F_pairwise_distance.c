// PTO Program: F_pairwise_distance
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_pairwise_distance
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,088 bytes (1.1 KB)
//   Total capacity (w/ reuse): 832 bytes (0.8 KB)
//   Reuse savings:            256 bytes (23.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  2,   3]           -
//   result               8x1        f32        32   [  5,   6]           -
//   row_sum              8x1        f32        32   [  4,   5]           -
//   sq_diff              8x8        f32       256   [  3,   4]           <- x1
//   x1                   8x8        f32       256   [  0,   2]           -
//   x2                   8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   sq_diff reuses buffer of x1
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_pairwise_distance(float* input1, float* input2, float* output) {
    float x1[8][8];
    float x2[8][8];
    float diff[8][8];
    float sq_diff[8][8];
    float row_sum[8][1];
    float result[8][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (4 ops): x1=TLOAD(input1,0,0); x2=TLOAD(input2,0,0); diff=TSUB(x1,x2); sq_diff=TMUL(diff,diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input1[_row * 8 + _col]);
            vst1q_f32(&x1[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input2[_row * 8 + _col]);
            vst1q_f32(&x2[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x1[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr4 = vsubq_f32(_v2, _v3);
            vst1q_f32(&diff[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&sq_diff[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1[_row][_col] = input1[_row * 8 + _col];
            x2[_row][_col] = input2[_row * 8 + _col];
            diff[_row][_col] = x1[_row][_col] - x2[_row][_col];
            sq_diff[_row][_col] = diff[_row][_col] * diff[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(sq_diff)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += sq_diff[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TSQRT(row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr9 = vsqrtq_f32(_v8);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = sqrtf(row_sum[_row][_col]);
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_pairwise_distance"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input1",
    "input2",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(32),
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
    F_pairwise_distance((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER