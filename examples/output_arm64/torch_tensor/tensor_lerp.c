// PTO Program: tensor_lerp
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_lerp
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            512 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  2,   3]           -
//   end                  8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  4,   5]           <- diff
//   scaled               8x8        f32       256   [  3,   4]           <- end
//   self                 8x8        f32       256   [  0,   4]           -
//
// BUFFER REUSE MAP:
//   scaled reuses buffer of end
//   result reuses buffer of diff
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_lerp(float* input_self, float* input_end, float* output) {
    float self[8][8];
    float end[8][8];
    float diff[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): self=TLOAD(input_self,0,0); end=TLOAD(input_end,0,0); diff=TSUB(end,self); scaled=TMULS(diff,0.5f); result=TADD(self,scaled); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_end[_row * 8 + _col]);
            vst1q_f32(&end[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&end[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr5 = vsubq_f32(_v3, _v4);
            vst1q_f32(&diff[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v6, _vs0);
            vst1q_f32(&scaled[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v8, _v9);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            end[_row][_col] = input_end[_row * 8 + _col];
            diff[_row][_col] = end[_row][_col] - self[_row][_col];
            scaled[_row][_col] = diff[_row][_col] * 0.5f;
            result[_row][_col] = self[_row][_col] + scaled[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_lerp"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_self",
    "input_end",
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
    tensor_lerp((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER