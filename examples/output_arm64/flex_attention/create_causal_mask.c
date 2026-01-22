// PTO Program: create_causal_mask
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: create_causal_mask
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 512 bytes (0.5 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mask                 8x8        f32       256   [  0,   2]           -
//   ones                 8x8        f32       256   [  1,  -1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void create_causal_mask(float* mask_mem) {
    float mask[8][8];
    float ones[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): mask=TEXPANDS(-1000000000.0f); ones=TEXPANDS(0.0f); mask_mem=TSTORE(mask,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1000000000.0f);
    float32x4_t _vs1 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&mask[_row][_col], _vs0);
            vst1q_f32(&ones[_row][_col], _vs1);
            float32x4_t _vs2 = vld1q_f32(&mask[_row][_col]);
            vst1q_f32(&mask_mem[_row * 8 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            mask[_row][_col] = -1000000000.0f;
            ones[_row][_col] = 0.0f;
            mask_mem[_row * 8 + _col] = mask[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "create_causal_mask"; }
enum { kPtoNumMemrefs = 1 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "mask_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
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
    create_causal_mask((float*)args[0]);
}
#endif  // PTO_CPU_SMOKE_RUNNER