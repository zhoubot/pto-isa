// PTO Program: nn_Embedding
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Embedding
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 4,352 bytes (4.2 KB)
//   Total capacity (w/ reuse): 4,352 bytes (4.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   indices_onehot       8x64       f32      2048   [  0,  -1]           -
//   result               8x8        f32       256   [  2,   3]           -
//   weight               64x8       f32      2048   [  1,  -1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_Embedding(float* indices_mem, float* weight_mem, float* output) {
    float indices_onehot[8][64];
    float weight[64][8];
    float result[8][8];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): indices_onehot=TLOAD(indices_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&indices_mem[_row * 64 + _col]);
            vst1q_f32(&indices_onehot[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            indices_onehot[_row][_col] = indices_mem[_row * 64 + _col];
        }
    }

    // FUSED LOOP (1 ops): weight=TLOAD(weight_mem,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&weight_mem[_row * 8 + _col]);
            vst1q_f32(&weight[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            weight[_row][_col] = weight_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: result = indices_onehot @ weight
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += indices_onehot[_i][_k] * weight[_k][_j];}
            result[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_Embedding"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "indices_mem",
    "weight_mem",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(2048),
    (size_t)(2048),
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
    nn_Embedding((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER