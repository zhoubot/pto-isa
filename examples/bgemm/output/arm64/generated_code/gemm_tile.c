// PTO Program: gemm_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: gemm_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 81,920 bytes (80.0 KB)
//   Total capacity (w/ reuse): 81,920 bytes (80.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void gemm_tile(float* A, float* B, float* C) {
    float a[64][64];
    float b[64][128];
    float c[64][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(A[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 64; _col++) {
            a[_row][_col] = A[_row * 64 + _col];
        }}

    // TLOAD: b = load(B[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            b[_row][_col] = B[_row * 128 + _col];
        }}

    // TMATMUL: c = a @ b
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += a[_i][_k] * b[_k][_j];}
            c[_i][_j] = _sum;}}

    // TSTORE: store(c) -> C[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            C[_row * 128 + _col] = c[_row][_col];
        }}

}