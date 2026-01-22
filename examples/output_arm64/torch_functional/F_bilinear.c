// PTO Program: F_bilinear
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_bilinear
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   output               8x8        f32       256   [  4,   5]           -
//   temp                 8x8        f32       256   [  3,   4]           -
//   weight               8x8        f32       256   [  2,  -1]           -
//   x1                   8x8        f32       256   [  0,  -1]           -
//   x2                   8x8        f32       256   [  1,   4]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_bilinear(float* input1, float* input2, float* weight_mem, float* output_mem) {
    float x1[8][8];
    float x2[8][8];
    float weight[8][8];
    float temp[8][8];
    float output[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (3 ops): x1=TLOAD(input1,0,0); x2=TLOAD(input2,0,0); weight=TLOAD(weight_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input1[_row * 8 + _col]);
            vst1q_f32(&x1[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input2[_row * 8 + _col]);
            vst1q_f32(&x2[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&weight_mem[_row * 8 + _col]);
            vst1q_f32(&weight[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1[_row][_col] = input1[_row * 8 + _col];
            x2[_row][_col] = input2[_row * 8 + _col];
            weight[_row][_col] = weight_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: temp = x1 @ weight
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += x1[_i][_k] * weight[_k][_j];}
            temp[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): output=TMUL(temp,x2); output_mem=TSTORE(output,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&temp[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&output[_row][_col], _vr5);
            float32x4_t _vs6 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row][_col] = temp[_row][_col] * x2[_row][_col];
            output_mem[_row * 8 + _col] = output[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_bilinear"; }
enum { kPtoNumMemrefs = 4 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input1",
    "input2",
    "weight_mem",
    "output_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
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
    F_bilinear((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3]);
}
#endif  // PTO_CPU_SMOKE_RUNNER