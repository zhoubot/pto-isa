// PTO Program: op_rowexpanddiv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: op_rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 131,584 bytes (128.5 KB)
//   Total capacity (w/ reuse): 131,584 bytes (128.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   row                  128x1      f32       512   [  1,   2]           -
//   x                    128x128    f32     65536   [  0,   2]           -
//   y                    128x128    f32     65536   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void op_rowexpanddiv(float* input, float* row_vals, float* output) {
    float x[128][128];
    float row[128][1];
    float y[128][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): row=TLOAD(row_vals,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&row_vals[_row * 1 + _col]);
            vst1q_f32(&row[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row[_row][_col] = row_vals[_row * 1 + _col];
        }
    }

    // FUSED LOOP (2 ops): y=TROWEXPANDDIV(x,row); output=TSTORE(y,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(row[_row][0]);
            float32x4_t _vr3 = vdivq_f32(_v02, _vb4);
            vst1q_f32(&y[_row][_col], _vr3);
            float32x4_t _vs5 = vld1q_f32(&y[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            y[_row][_col] = x[_row][_col] / row[_row][0];
            output[_row * 128 + _col] = y[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "op_rowexpanddiv"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "row_vals",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(65536),
    (size_t)(512),
    (size_t)(65536),
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
    op_rowexpanddiv((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER