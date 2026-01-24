// PTO Program: tile_add
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_add
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_add(float* A, float* B, float* C) {
    float a[64][128];
    float b[64][128];
    float c[64][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(A[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            a[_row][_col] = A[_row * 128 + _col];
        }}

    // TLOAD: b = load(B[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            b[_row][_col] = B[_row * 128 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            c[_row][_col] = a[_row][_col] + b[_row][_col];
        }}
    }

    // TSTORE: store(c) -> C[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            C[_row * 128 + _col] = c[_row][_col];
        }}

}