// PTO Program: nn_Embedding
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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