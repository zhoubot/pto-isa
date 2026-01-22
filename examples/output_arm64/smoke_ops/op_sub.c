// PTO Program: op_sub
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: op_sub
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 196,608 bytes (192.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   a                    128x128    f32     65536   [  0,   2]           -
//   b                    128x128    f32     65536   [  1,   2]           -
//   y                    128x128    f32     65536   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void op_sub(float* input0, float* input1, float* output) {
    float a[128][128];
    float b[128][128];
    float y[128][128];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (4 ops): a=TLOAD(input0,0,0); b=TLOAD(input1,0,0); y=TSUB(a,b); output=TSTORE(y,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input0[_row * 128 + _col]);
            vst1q_f32(&a[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input1[_row * 128 + _col]);
            vst1q_f32(&b[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&a[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&b[_row][_col]);
            float32x4_t _vr4 = vsubq_f32(_v2, _v3);
            vst1q_f32(&y[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&y[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            a[_row][_col] = input0[_row * 128 + _col];
            b[_row][_col] = input1[_row * 128 + _col];
            y[_row][_col] = a[_row][_col] - b[_row][_col];
            output[_row * 128 + _col] = y[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "op_sub"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input0",
    "input1",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(65536),
    (size_t)(65536),
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
    op_sub((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER