// PTO Program: residual_add_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: residual_add_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   residual             64x128     f32     32768   [  1,   2]           -
//   result               64x128     f32     32768   [  2,   3]           -
//   x                    64x128     f32     32768   [  0,   2]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void residual_add_tile_64(float* input, float* input_residual, float* output) {
    float x[64][128];
    float residual[64][128];
    float result[64][128];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (4 ops): x=TLOAD(input,0,0); residual=TLOAD(input_residual,0,0); result=TADD(x,residual); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_residual[_row * 128 + _col]);
            vst1q_f32(&residual[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&residual[_row][_col]);
            float32x4_t _vr4 = vaddq_f32(_v2, _v3);
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
            residual[_row][_col] = input_residual[_row * 128 + _col];
            result[_row][_col] = x[_row][_col] + residual[_row][_col];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}