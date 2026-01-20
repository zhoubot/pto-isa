// PTO Program: tensor_clamp
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_clamp(float* input, float* output) {
    float self[8][8];
    float min_tile[8][8];
    float max_tile[8][8];
    float clamp_low[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): self=TLOAD(input,0,0); min_tile=TEXPANDS(-1.0f); max_tile=TEXPANDS(1.0f); clamp_low=TMAX(self,min_tile); result=TMIN(clamp_low,max_tile); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl2);
            vst1q_f32(&min_tile[_row][_col], _vs0);
            vst1q_f32(&max_tile[_row][_col], _vs1);
            float32x4_t _v3 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&min_tile[_row][_col]);
            float32x4_t _vr5 = vmaxq_f32(_v3, _v4);
            vst1q_f32(&clamp_low[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&clamp_low[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&max_tile[_row][_col]);
            float32x4_t _vr8 = vminq_f32(_v6, _v7);
            vst1q_f32(&result[_row][_col], _vr8);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            min_tile[_row][_col] = -1.0f;
            max_tile[_row][_col] = 1.0f;
            clamp_low[_row][_col] = self[_row][_col] + min_tile[_row][_col];
            result[_row][_col] = clamp_low[_row][_col] + max_tile[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}