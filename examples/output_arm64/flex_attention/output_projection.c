// PTO Program: output_projection
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void output_projection(float* attn_mem, float* WO_mem, float* output_mem) {
    float attn_out[8][8];
    float W_O[8][64];
    float output[8][64];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): attn_out=TLOAD(attn_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&attn_mem[_row * 8 + _col]);
            vst1q_f32(&attn_out[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            attn_out[_row][_col] = attn_mem[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): W_O=TLOAD(WO_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&WO_mem[_row * 64 + _col]);
            vst1q_f32(&W_O[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            W_O[_row][_col] = WO_mem[_row * 64 + _col];
        }
    }

    // TMATMUL: output = attn_out @ W_O
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 64; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += attn_out[_i][_k] * W_O[_k][_j];}
            output[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output_mem=TSTORE(output,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 64 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            output_mem[_row * 64 + _col] = output[_row][_col];
        }
    }

}