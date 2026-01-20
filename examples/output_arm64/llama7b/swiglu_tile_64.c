// PTO Program: swiglu_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            131,072 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_gate         64x128     f32     32768   [  3,   4]           -
//   gate                 64x128     f32     32768   [  0,   6]           -
//   gate_silu            64x128     f32     32768   [  6,   7]           <- one_plus_exp
//   neg_gate             64x128     f32     32768   [  2,   3]           -
//   one_plus_exp         64x128     f32     32768   [  4,   5]           <- neg_gate
//   result               64x128     f32     32768   [  7,   8]           <- gate
//   sigmoid_gate         64x128     f32     32768   [  5,   6]           <- exp_neg_gate
//   up                   64x128     f32     32768   [  1,   7]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_gate
//   sigmoid_gate reuses buffer of exp_neg_gate
//   gate_silu reuses buffer of one_plus_exp
//   result reuses buffer of gate
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void swiglu_tile_64(float* input_gate, float* input_up, float* output) {
    float gate[64][128];
    float up[64][128];
    float neg_gate[64][128];
    float exp_neg_gate[64][128];
    float one_plus_exp[64][128];
    float sigmoid_gate[64][128];
    float gate_silu[64][128];
    float result[64][128];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (9 ops): gate=TLOAD(input_gate,0,0); up=TLOAD(input_up,0,0); neg_gate=TNEG(gate); exp_neg_gate=TEXP(neg_gate); one_plus_exp=TADDS(exp_neg_gate,1.0f); sigmoid_gate=TRECIP(one_plus_exp); gate_silu=TMUL(gate,sigmoid_gate); result=TMUL(gate_silu,up); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_gate[_row * 128 + _col]);
            vst1q_f32(&gate[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_up[_row * 128 + _col]);
            vst1q_f32(&up[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&gate[_row][_col]);
            float32x4_t _vr4 = vnegq_f32(_v3);
            vst1q_f32(&neg_gate[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&neg_gate[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&exp_neg_gate[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&exp_neg_gate[_row][_col]);
            float32x4_t _vr8 = vaddq_f32(_v7, _vs0);
            vst1q_f32(&one_plus_exp[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&one_plus_exp[_row][_col]);
            float32x4_t _vr10 = _v9;
            vst1q_f32(&sigmoid_gate[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&gate[_row][_col]);
            float32x4_t _v12 = vld1q_f32(&sigmoid_gate[_row][_col]);
            float32x4_t _vr13 = vmulq_f32(_v11, _v12);
            vst1q_f32(&gate_silu[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&gate_silu[_row][_col]);
            float32x4_t _v15 = vld1q_f32(&up[_row][_col]);
            float32x4_t _vr16 = vmulq_f32(_v14, _v15);
            vst1q_f32(&result[_row][_col], _vr16);
            float32x4_t _vs17 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs17);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            gate[_row][_col] = input_gate[_row * 128 + _col];
            up[_row][_col] = input_up[_row * 128 + _col];
            neg_gate[_row][_col] = -gate[_row][_col];
            exp_neg_gate[_row][_col] = expf(neg_gate[_row][_col]);
            one_plus_exp[_row][_col] = exp_neg_gate[_row][_col] + 1.0f;
            sigmoid_gate[_row][_col] = 1.0f / one_plus_exp[_row][_col];
            gate_silu[_row][_col] = gate[_row][_col] * sigmoid_gate[_row][_col];
            result[_row][_col] = gate_silu[_row][_col] * up[_row][_col];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}