// PTO Program: rope_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            65,536 bytes (33.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   cos_pos              64x128     f32     32768   [  1,   3]           -
//   result               64x128     f32     32768   [  5,   6]           <- x
//   sin_pos              64x128     f32     32768   [  2,   4]           -
//   x                    64x128     f32     32768   [  0,   4]           -
//   x_cos                64x128     f32     32768   [  3,   5]           -
//   x_sin                64x128     f32     32768   [  4,   5]           <- cos_pos
//
// BUFFER REUSE MAP:
//   x_sin reuses buffer of cos_pos
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rope_tile_64(float* input, float* cos_cache, float* sin_cache, float* output) {
    float x[64][128];
    float cos_pos[64][128];
    float sin_pos[64][128];
    float x_cos[64][128];
    float x_sin[64][128];
    float result[64][128];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); cos_pos=TLOAD(cos_cache,0,0); sin_pos=TLOAD(sin_cache,0,0); x_cos=TMUL(x,cos_pos); x_sin=TMUL(x,sin_pos); result=TADD(x_cos,x_sin); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&cos_cache[_row * 128 + _col]);
            vst1q_f32(&cos_pos[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&sin_cache[_row * 128 + _col]);
            vst1q_f32(&sin_pos[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&cos_pos[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&x_cos[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&sin_pos[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v6, _v7);
            vst1q_f32(&x_sin[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&x_cos[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&x_sin[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v9, _v10);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
            cos_pos[_row][_col] = cos_cache[_row * 128 + _col];
            sin_pos[_row][_col] = sin_cache[_row * 128 + _col];
            x_cos[_row][_col] = x[_row][_col] * cos_pos[_row][_col];
            x_sin[_row][_col] = x[_row][_col] * sin_pos[_row][_col];
            result[_row][_col] = x_cos[_row][_col] + x_sin[_row][_col];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}