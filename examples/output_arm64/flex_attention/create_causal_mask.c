// PTO Program: create_causal_mask
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void create_causal_mask(float* mask_mem) {
    float mask[8][8];
    float ones[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): mask=TEXPANDS(-1000000000.0f); ones=TEXPANDS(0.0f); mask_mem=TSTORE(mask,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1000000000.0f);
    float32x4_t _vs1 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&mask[_row][_col], _vs0);
            vst1q_f32(&ones[_row][_col], _vs1);
            float32x4_t _vs2 = vld1q_f32(&mask[_row][_col]);
            vst1q_f32(&mask_mem[_row * 8 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            mask[_row][_col] = -1000000000.0f;
            ones[_row][_col] = 0.0f;
            mask_mem[_row * 8 + _col] = mask[_row][_col];
        }
    }

}