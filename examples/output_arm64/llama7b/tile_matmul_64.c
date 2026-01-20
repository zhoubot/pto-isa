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