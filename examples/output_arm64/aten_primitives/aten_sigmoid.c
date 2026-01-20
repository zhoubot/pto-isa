// PTO Program: aten_sigmoid
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_sigmoid(float* input, float* output) {
    float x[8][8];
    float t1[8][8];
    float t2[8][8];
    float t3[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): x=TLOAD(input,0,0); t1=TNEG(x); t2=TEXP(t1); t3=TADDS(t2,1.0f); result=TRECIP(t3); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&t1[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&t1[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&t2[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&t2[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&t3[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&t3[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            t1[_row][_col] = -x[_row][_col];
            t2[_row][_col] = expf(t1[_row][_col]);
            t3[_row][_col] = t2[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / t3[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}