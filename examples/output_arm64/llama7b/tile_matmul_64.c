// PTO Program: tile_matmul_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_matmul_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   a                    64x128     f32     32768   [  0,  -1]           -
//   b                    128x128    f32     65536   [  1,  -1]           -
//   c                    64x128     f32     32768   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_matmul_64(float* input_a, float* input_b, float* output) {
    float a[64][128];
    float b[128][128];
    float c[64][128];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): a=TLOAD(input_a,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_a[_row * 128 + _col]);
            vst1q_f32(&a[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            a[_row][_col] = input_a[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): b=TLOAD(input_b,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_b[_row * 128 + _col]);
            vst1q_f32(&b[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            b[_row][_col] = input_b[_row * 128 + _col];
        }
    }

    // TMATMUL: c = a @ b
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += a[_i][_k] * b[_k][_j];}
            c[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(c,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&c[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            output[_row * 128 + _col] = c[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tile_matmul_64"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_a",
    "input_b",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(65536),
    (size_t)(32768),
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
    tile_matmul_64((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER