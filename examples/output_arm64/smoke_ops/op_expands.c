// PTO Program: op_expands
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: op_expands
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     1
//   Total capacity (no reuse): 65,536 bytes (64.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   y                    128x128    f32     65536   [  0,   1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void op_expands(float* output) {
    float y[128][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (2 ops): y=TEXPANDS(1.0f); output=TSTORE(y,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            vst1q_f32(&y[_row][_col], _vs0);
            float32x4_t _vs1 = vld1q_f32(&y[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs1);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            y[_row][_col] = 1.0f;
            output[_row * 128 + _col] = y[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "op_expands"; }
enum { kPtoNumMemrefs = 1 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(65536),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
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
    op_expands((float*)args[0]);
}
#endif  // PTO_CPU_SMOKE_RUNNER