// PTO Program: attention_output_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_output_tile_64
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
//   result               64x128     f32     32768   [  2,   3]           -
//   v                    128x128    f32     65536   [  1,  -1]           -
//   weights              64x128     f32     32768   [  0,  -1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_output_tile_64(float* input_weights, float* input_v, float* output) {
    float weights[64][128];
    float v[128][128];
    float result[64][128];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): weights=TLOAD(input_weights,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_weights[_row * 128 + _col]);
            vst1q_f32(&weights[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            weights[_row][_col] = input_weights[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): v=TLOAD(input_v,0,0)
    for (int _row = 0; _row < 128; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_v[_row * 128 + _col]);
            vst1q_f32(&v[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            v[_row][_col] = input_v[_row * 128 + _col];
        }
    }

    // TMATMUL: result = weights @ v
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += weights[_i][_k] * v[_k][_j];}
            result[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}