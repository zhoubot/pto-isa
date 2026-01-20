// PTO Program: prims_pow
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void prims_pow(float* input_base, float* input_exp, float* output) {
    float base[8][8];
    float exp[8][8];
    float log_base[8][8];
    float product[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): base=TLOAD(input_base,0,0); exp=TLOAD(input_exp,0,0); log_base=TLOG(base); product=TMUL(exp,log_base); result=TEXP(product); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_base[_row * 8 + _col]);
            vst1q_f32(&base[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_exp[_row * 8 + _col]);
            vst1q_f32(&exp[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&base[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_base[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&exp[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&log_base[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&product[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&product[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&result[_row][_col], _vr8);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            base[_row][_col] = input_base[_row * 8 + _col];
            exp[_row][_col] = input_exp[_row * 8 + _col];
            log_base[_row][_col] = logf(base[_row][_col]);
            product[_row][_col] = exp[_row][_col] * log_base[_row][_col];
            result[_row][_col] = expf(product[_row][_col]);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}