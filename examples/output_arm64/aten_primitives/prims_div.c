// PTO Program: prims_div
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: prims_div
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 49,152 bytes (48.0 KB)
//   Total capacity (w/ reuse): 49,152 bytes (48.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               1x4096     f32     16384   [  5,  13]           -
//   x                    1x4096     f32     16384   [  3,  12]           -
//   y                    1x4096     f32     16384   [  4,  12]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void prims_div(float* input_x, float* input_y, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    float x[1][4096];
    float y[1][4096];
    float result[1][4096];

    // Loop fusion: 6 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (4 ops): x=TLOAD(input_x,tile_idx,0); y=TLOAD(input_y,tile_idx,0); result=TDIV(x,y); output=TSTORE(result,tile_idx,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl0 = vld1q_f32(&input_x[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl0);
                float32x4_t _vl1 = vld1q_f32(&input_y[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&y[_row][_col], _vl1);
                float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v3 = vld1q_f32(&y[_row][_col]);
                float32x4_t _vr4 = vdivq_f32(_v2, _v3);
                vst1q_f32(&result[_row][_col], _vr4);
                float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs5);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input_x[(tile_idx) * 4096 + _row * 4096 + _col];
                y[_row][_col] = input_y[(tile_idx) * 4096 + _row * 4096 + _col];
                result[_row][_col] = x[_row][_col] / y[_row][_col];
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (4 ops): x=TLOAD(input_x,num_full_tiles,0); y=TLOAD(input_y,num_full_tiles,0); result=TDIV(x,y); output=TSTORE(result,num_full_tiles,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl6 = vld1q_f32(&input_x[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl6);
                float32x4_t _vl7 = vld1q_f32(&input_y[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&y[_row][_col], _vl7);
                float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v9 = vld1q_f32(&y[_row][_col]);
                float32x4_t _vr10 = vdivq_f32(_v8, _v9);
                vst1q_f32(&result[_row][_col], _vr10);
                float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs11);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input_x[(num_full_tiles) * 4096 + _row * 4096 + _col];
                y[_row][_col] = input_y[(num_full_tiles) * 4096 + _row * 4096 + _col];
                result[_row][_col] = x[_row][_col] / y[_row][_col];
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "prims_div"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_x",
    "input_y",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
    (size_t)(16384),
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
    prims_div((float*)args[0], (float*)args[1], (float*)args[2], (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER