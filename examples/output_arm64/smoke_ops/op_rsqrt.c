// PTO Program: op_rsqrt
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: op_rsqrt
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   x                    128x128    f32     65536   [  0,   1]           -
//   y                    128x128    f32     65536   [  1,   2]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void op_rsqrt(float* input, float* output) {
    float x[128][128];
    float y[128][128];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(input,0,0); y=TRSQRT(x); output=TSTORE(y,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = vrsqrteq_f32(_v1);
            vst1q_f32(&y[_row][_col], _vr2);
            float32x4_t _vs3 = vld1q_f32(&y[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs3);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
            y[_row][_col] = 1.0f / sqrtf(x[_row][_col]);
            output[_row * 128 + _col] = y[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "op_rsqrt"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(65536),
    (size_t)(65536),
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
    op_rsqrt((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER