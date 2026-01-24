// PTO Program: tile_copy
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_copy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     1
//   Total capacity (no reuse): 32,768 bytes (32.0 KB)
//   Total capacity (w/ reuse): 32,768 bytes (32.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_copy(float* A, float* C) {
    float a[64][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(A[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            a[_row][_col] = A[_row * 128 + _col];
        }}

    // TSTORE: store(a) -> C[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            C[_row * 128 + _col] = a[_row][_col];
        }}

}