// PTO Program: attention_score_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_score_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 163,840 bytes (160.0 KB)
//   Total capacity (w/ reuse): 163,840 bytes (160.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k_t                  128x128    f32     65536   [  1,  -1]           -
//   q                    64x128     f32     32768   [  0,  -1]           -
//   scaled_scores        64x128     f32     32768   [  4,   5]           -
//   scores               64x128     f32     32768   [  2,   4]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_score_tile_64(float* input_q, float* input_kt, float* output) {
    float q[64][128];
    float k_t[128][128];
    float scores[64][128];
    float scaled_scores[64][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): q=TLOAD(input_q,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_q[_row * 128 + _col]);
            vst1q_f32(&q[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            q[_row][_col] = input_q[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): k_t=TLOAD(input_kt,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_kt[_row * 128 + _col]);
            vst1q_f32(&k_t[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            k_t[_row][_col] = input_kt[_row * 128 + _col];
        }
    }

    // TMATMUL: scores = q @ k_t
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            scores[_i][_j] = _sum;}}

    int scale = 0.08838834764831843;

    // FUSED LOOP (2 ops): scaled_scores=TMULS(scores,scalef); output=TSTORE(scaled_scores,0,0)
    float32x4_t _vs2 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v3, _vs2);
            vst1q_f32(&scaled_scores[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&scaled_scores[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            scaled_scores[_row][_col] = scores[_row][_col] * scalef;
            output[_row * 128 + _col] = scaled_scores[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "attention_score_tile_64"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_q",
    "input_kt",
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
    attention_score_tile_64((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER