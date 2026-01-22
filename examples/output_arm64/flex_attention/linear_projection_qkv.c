// PTO Program: linear_projection_qkv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: linear_projection_qkv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 8,960 bytes (8.8 KB)
//   Total capacity (w/ reuse): 8,960 bytes (8.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  5,   8]           -
//   Q                    8x8        f32       256   [  4,   7]           -
//   V                    8x8        f32       256   [  6,   9]           -
//   W_K                  64x8       f32      2048   [  2,  -1]           -
//   W_Q                  64x8       f32      2048   [  1,  -1]           -
//   W_V                  64x8       f32      2048   [  3,  -1]           -
//   X                    8x64       f32      2048   [  0,  -1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "linear_projection_qkv"; }
enum { kPtoNumMemrefs = 7 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "X_mem",
    "WQ_mem",
    "WK_mem",
    "WV_mem",
    "Q_mem",
    "K_mem",
    "V_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(2048),
    (size_t)(2048),
    (size_t)(2048),
    (size_t)(2048),
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    linear_projection_qkv((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5], (float*)args[6]);
}
#endif  // PTO_CPU_SMOKE_RUNNER