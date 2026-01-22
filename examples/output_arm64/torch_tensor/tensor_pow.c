// PTO Program: tensor_pow
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_pow
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            512 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_self             8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  3,   4]           <- log_self
//   scaled               8x8        f32       256   [  2,   3]           <- self
//   self                 8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   scaled reuses buffer of self
//   result reuses buffer of log_self
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_pow(float* input, float* output) {
    float self[8][8];
    float log_self[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): self=TLOAD(input,0,0); log_self=TLOG(self); scaled=TMULS(log_self,0.5f); result=TEXP(scaled); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_self[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&log_self[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs0);
            vst1q_f32(&scaled[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _vs8 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs8);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            log_self[_row][_col] = logf(self[_row][_col]);
            scaled[_row][_col] = log_self[_row][_col] * 0.5f;
            result[_row][_col] = expf(scaled[_row][_col]);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_pow"; }
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
    tensor_pow((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER