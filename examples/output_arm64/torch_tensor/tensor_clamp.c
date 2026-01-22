// PTO Program: tensor_clamp
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_clamp
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            256 bytes (20.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   clamp_low            8x8        f32       256   [  3,   4]           -
//   max_tile             8x8        f32       256   [  2,   4]           -
//   min_tile             8x8        f32       256   [  1,   3]           -
//   result               8x8        f32       256   [  4,   5]           <- self
//   self                 8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of self
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_clamp(float* input, float* output) {
    float self[8][8];
    float min_tile[8][8];
    float max_tile[8][8];
    float clamp_low[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): self=TLOAD(input,0,0); min_tile=TEXPANDS(-1.0f); max_tile=TEXPANDS(1.0f); clamp_low=TMAX(self,min_tile); result=TMIN(clamp_low,max_tile); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl2);
            vst1q_f32(&min_tile[_row][_col], _vs0);
            vst1q_f32(&max_tile[_row][_col], _vs1);
            float32x4_t _v3 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&min_tile[_row][_col]);
            float32x4_t _vr5 = vmaxq_f32(_v3, _v4);
            vst1q_f32(&clamp_low[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&clamp_low[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&max_tile[_row][_col]);
            float32x4_t _vr8 = vminq_f32(_v6, _v7);
            vst1q_f32(&result[_row][_col], _vr8);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            min_tile[_row][_col] = -1.0f;
            max_tile[_row][_col] = 1.0f;
            clamp_low[_row][_col] = self[_row][_col] + min_tile[_row][_col];
            result[_row][_col] = clamp_low[_row][_col] + max_tile[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_clamp"; }
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
    tensor_clamp((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER