// PTO Program: F_threshold
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_threshold
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  3,   4]           -
//   thresh_tile          8x8        f32       256   [  1,   3]           -
//   value_tile           8x8        f32       256   [  2,  -1]           -
//   x                    8x8        f32       256   [  0,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_threshold(float* input, float* output) {
    float x[8][8];
    float thresh_tile[8][8];
    float value_tile[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): x=TLOAD(input,0,0); thresh_tile=TEXPANDS(0.0f); value_tile=TEXPANDS(0.0f); result=TMAX(x,thresh_tile); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            vst1q_f32(&thresh_tile[_row][_col], _vs0);
            vst1q_f32(&value_tile[_row][_col], _vs0);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&thresh_tile[_row][_col]);
            float32x4_t _vr4 = vmaxq_f32(_v2, _v3);
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            thresh_tile[_row][_col] = 0.0f;
            value_tile[_row][_col] = 0.0f;
            result[_row][_col] = x[_row][_col] + thresh_tile[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_threshold"; }
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
    F_threshold((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER