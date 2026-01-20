// PTO Program: linear_projection_qkv
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void linear_projection_qkv(float* X_mem, float* WQ_mem, float* WK_mem, float* WV_mem, float* Q_mem, float* K_mem, float* V_mem) {
    float X[8][64];
    float W_Q[64][8];
    float W_K[64][8];
    float W_V[64][8];
    float Q[8][8];
    float K[8][8];
    float V[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (1 ops): X=TLOAD(X_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&X_mem[_row * 64 + _col]);
            vst1q_f32(&X[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            X[_row][_col] = X_mem[_row * 64 + _col];
        }
    }

    // FUSED LOOP (3 ops): W_Q=TLOAD(WQ_mem,0,0); W_K=TLOAD(WK_mem,0,0); W_V=TLOAD(WV_mem,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&WQ_mem[_row * 8 + _col]);
            vst1q_f32(&W_Q[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&WK_mem[_row * 8 + _col]);
            vst1q_f32(&W_K[_row][_col], _vl2);
            float32x4_t _vl3 = vld1q_f32(&WV_mem[_row * 8 + _col]);
            vst1q_f32(&W_V[_row][_col], _vl3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            W_Q[_row][_col] = WQ_mem[_row * 8 + _col];
            W_K[_row][_col] = WK_mem[_row * 8 + _col];
            W_V[_row][_col] = WV_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: Q = X @ W_Q
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += X[_i][_k] * W_Q[_k][_j];}
            Q[_i][_j] = _sum;}}

    // TMATMUL: K = X @ W_K
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += X[_i][_k] * W_K[_k][_j];}
            K[_i][_j] = _sum;}}

    // TMATMUL: V = X @ W_V
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += X[_i][_k] * W_V[_k][_j];}
            V[_i][_j] = _sum;}}

    // FUSED LOOP (3 ops): Q_mem=TSTORE(Q,0,0); K_mem=TSTORE(K,0,0); V_mem=TSTORE(V,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs4 = vld1q_f32(&Q[_row][_col]);
            vst1q_f32(&Q_mem[_row * 8 + _col], _vs4);
            float32x4_t _vs5 = vld1q_f32(&K[_row][_col]);
            vst1q_f32(&K_mem[_row * 8 + _col], _vs5);
            float32x4_t _vs6 = vld1q_f32(&V[_row][_col]);
            vst1q_f32(&V_mem[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            Q_mem[_row * 8 + _col] = Q[_row][_col];
            K_mem[_row * 8 + _col] = K[_row][_col];
            V_mem[_row * 8 + _col] = V[_row][_col];
        }
    }

}