// PTO Program: tile_muls_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_muls_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 65,536 bytes (64.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   a                    64x128     f32     32768   [  0,   1]           -
//   result               64x128     f32     32768   [  1,   2]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_muls_64(float* input, float* output, float scale) {
    float a[64][128];
    float result[64][128];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): a=TLOAD(input,0,0); result=TMULS(a,scalef); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&a[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&a[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v2, _vs0);
            vst1q_f32(&result[_row][_col], _vr3);
            float32x4_t _vs4 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs4);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            a[_row][_col] = input[_row * 128 + _col];
            result[_row][_col] = a[_row][_col] * scalef;
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}