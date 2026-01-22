// PTO Program: tensor_xlogy
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_xlogy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            256 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_other            8x8        f32       256   [  2,   3]           -
//   other                8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  3,   4]           <- other
//   self                 8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of other
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_xlogy(float* input_self, float* input_other, float* output) {
    float self[8][8];
    float other[8][8];
    float log_other[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): self=TLOAD(input_self,0,0); other=TLOAD(input_other,0,0); log_other=TLOG(other); result=TMUL(self,log_other); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_other[_row * 8 + _col]);
            vst1q_f32(&other[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&other[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_other[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&log_other[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&result[_row][_col], _vr6);
            float32x4_t _vs7 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            other[_row][_col] = input_other[_row * 8 + _col];
            log_other[_row][_col] = logf(other[_row][_col]);
            result[_row][_col] = self[_row][_col] * log_other[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_xlogy"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_self",
    "input_other",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
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
    tensor_xlogy((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER