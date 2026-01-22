// PTO Program: rowexpanddiv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 544 bytes (0.5 KB)
//   Total capacity (w/ reuse): 544 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  2,   3]           -
//   row_vals             8x1        f32        32   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   2]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rowexpanddiv(float* input_x, float* input_row, float* output) {
    float x[8][8];
    float row_vals[8][1];
    float result[8][8];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input_x,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_x[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input_x[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): row_vals=TLOAD(input_row,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_row[_row * 1 + _col]);
            vst1q_f32(&row_vals[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_vals[_row][_col] = input_row[_row * 1 + _col];
        }
    }

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(x,row_vals); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(row_vals[_row][0]);
            float32x4_t _vr3 = vdivq_f32(_v02, _vb4);
            vst1q_f32(&result[_row][_col], _vr3);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] / row_vals[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "rowexpanddiv"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_x",
    "input_row",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(32),
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
    rowexpanddiv((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER