// PTO Program: tensor_prod
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_prod
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 552 bytes (0.5 KB)
//   Total capacity (w/ reuse): 552 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_self             8x8        f32       256   [  1,   2]           -
//   result               1x1        f32         4   [  4,   5]           -
//   row_sum              8x1        f32        32   [  2,   3]           -
//   self                 8x8        f32       256   [  0,   1]           -
//   total                1x1        f32         4   [  3,   4]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_prod(float* input, float* output) {
    float self[8][8];
    float log_self[8][8];
    float row_sum[8][1];
    float total[1][1];
    float result[1][1];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): self=TLOAD(input,0,0); log_self=TLOG(self)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&log_self[_row][_col], _vr2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            log_self[_row][_col] = logf(self[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(log_self)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += log_self[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TEXP(total); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&total[_row][_col]);
            float32x4_t _vr4 = _v3;
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = expf(total[_row][_col]);
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_prod"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(4),
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
    tensor_prod((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER