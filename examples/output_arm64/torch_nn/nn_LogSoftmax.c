// PTO Program: nn_LogSoftmax
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_LogSoftmax
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 832 bytes (0.8 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            256 bytes (30.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                8x8        f32       256   [  1,   2]           -
//   log_sum              8x1        f32        32   [  3,   4]           -
//   result               8x8        f32       256   [  4,   5]           <- exp_x
//   sum_exp              8x1        f32        32   [  2,   3]           -
//   x                    8x8        f32       256   [  0,   4]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of exp_x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_LogSoftmax(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float sum_exp[8][1];
    float log_sum[8][1];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): x=TLOAD(input,0,0); exp_x=TEXP(x)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&exp_x[_row][_col], _vr2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_x[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(sum_exp)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&sum_exp[_row][_col]);
            float32x4_t _vr4 = _v3;
            vst1q_f32(&log_sum[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(sum_exp[_row][_col]);
        }
    }

    // FUSED LOOP (2 ops): result=TROWEXPANDSUB(x,log_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v05 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb7 = vdupq_n_f32(log_sum[_row][0]);
            float32x4_t _vr6 = vsubq_f32(_v05, _vb7);
            vst1q_f32(&result[_row][_col], _vr6);
            float32x4_t _vs8 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs8);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] - log_sum[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_LogSoftmax"; }
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
    nn_LogSoftmax((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER