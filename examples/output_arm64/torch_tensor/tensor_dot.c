// PTO Program: tensor_dot
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_dot
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 772 bytes (0.8 KB)
//   Total capacity (w/ reuse): 772 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   other                1x64       f32       256   [  1,   2]           -
//   prod                 1x64       f32       256   [  2,   3]           -
//   result               1x1        f32         4   [  3,   4]           -
//   self                 1x64       f32       256   [  0,   2]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_dot(float* input_self, float* input_other, float* output) {
    float self[1][64];
    float other[1][64];
    float prod[1][64];
    float result[1][1];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): self=TLOAD(input_self,0,0); other=TLOAD(input_other,0,0); prod=TMUL(self,other)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 64 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_other[_row * 64 + _col]);
            vst1q_f32(&other[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&other[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&prod[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            self[_row][_col] = input_self[_row * 64 + _col];
            other[_row][_col] = input_other[_row * 64 + _col];
            prod[_row][_col] = self[_row][_col] * other[_row][_col];
        }
    }

    // TROWSUM: result = rowsum(prod)
    for (int _row = 0; _row < 1; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 64; _col++) {
            _sum += prod[_row][_col];
        }
        result[_row][0] = _sum;}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_dot"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_self",
    "input_other",
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
    tensor_dot((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER