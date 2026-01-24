// PTO Program: bgemm_dynamic
// Function Type: Orchestration (control flow only)
// Orchestration function - builds task graph using PTO runtime
#include "pto_runtime.h"
#include "pto_runtime.c"  // Include for standalone build
#include <string.h>  // For strcmp in main
#include <time.h>    // For benchmark timing

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void bgemm_dynamic(PTORuntime* rt, float* A, float* B, float* C, float* P0, float* P1, float* P2, int32_t seq_len, int32_t tile_rows, int32_t num_tiles, float zero) {

    // Loop fusion: 0 loop overheads saved

    // Binary-expanded loop: tile in [0, num_tiles), max_range=4096
    int tile_remaining_0 = num_tiles - 0;
    int tile_base_0 = 0;
    if (tile_remaining_0 >= 4096) {
        for (int tile = tile_base_0; tile < tile_base_0 + 4096; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 4096;
        tile_remaining_0 -= 4096;
    }
    if (tile_remaining_0 >= 2048) {
        for (int tile = tile_base_0; tile < tile_base_0 + 2048; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 2048;
        tile_remaining_0 -= 2048;
    }
    if (tile_remaining_0 >= 1024) {
        for (int tile = tile_base_0; tile < tile_base_0 + 1024; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 1024;
        tile_remaining_0 -= 1024;
    }
    if (tile_remaining_0 >= 512) {
        for (int tile = tile_base_0; tile < tile_base_0 + 512; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 512;
        tile_remaining_0 -= 512;
    }
    if (tile_remaining_0 >= 256) {
        for (int tile = tile_base_0; tile < tile_base_0 + 256; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 256;
        tile_remaining_0 -= 256;
    }
    if (tile_remaining_0 >= 128) {
        for (int tile = tile_base_0; tile < tile_base_0 + 128; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 128;
        tile_remaining_0 -= 128;
    }
    if (tile_remaining_0 >= 64) {
        for (int tile = tile_base_0; tile < tile_base_0 + 64; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 64;
        tile_remaining_0 -= 64;
    }
    if (tile_remaining_0 >= 32) {
        for (int tile = tile_base_0; tile < tile_base_0 + 32; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 32;
        tile_remaining_0 -= 32;
    }
    if (tile_remaining_0 >= 16) {
        for (int tile = tile_base_0; tile < tile_base_0 + 16; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 16;
        tile_remaining_0 -= 16;
    }
    if (tile_remaining_0 >= 8) {
        for (int tile = tile_base_0; tile < tile_base_0 + 8; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
        }
        tile_base_0 += 8;
        tile_remaining_0 -= 8;
    }
    // Residual loop for remaining < 8
    for (int tile = tile_base_0; tile < tile_base_0 + tile_remaining_0; tile += 1) {
    
            // Task 0: gemm_tile
            int32_t t0 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t0, A, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t0, B, 0 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t0, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: gemm_tile
            int32_t t1 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t1, A, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t1, B, 1 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t1, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: gemm_tile
            int32_t t2 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t2, A, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t2, B, 2 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t2, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: gemm_tile
            int32_t t3 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t3, A, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t3, B, 3 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t3, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: gemm_tile
            int32_t t4 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t4, A, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t4, B, 4 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t4, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: gemm_tile
            int32_t t5 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t5, A, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t5, B, 5 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t5, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
            // Task 6: gemm_tile
            int32_t t6 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t6, A, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t6, B, 6 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t6, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // Task 7: gemm_tile
            int32_t t7 = pto_task_alloc(rt, "gemm_tile", NULL, 81920, 81920, 1);
            pto_task_add_input(rt, t7, A, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t7, B, 7 * num_tiles + tile, 0, 32, 128);
            pto_task_add_input(rt, t7, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_submit(rt, t7);
    
    
            // Task 8: tile_add
            int32_t t8 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t8, P0, tile * 8 + 0, 0, 32, 128);
            pto_task_add_input(rt, t8, P0, tile * 8 + 1, 0, 32, 128);
            pto_task_add_input(rt, t8, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_submit(rt, t8);
    
    
            // Task 9: tile_add
            int32_t t9 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t9, P0, tile * 8 + 2, 0, 32, 128);
            pto_task_add_input(rt, t9, P0, tile * 8 + 3, 0, 32, 128);
            pto_task_add_input(rt, t9, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_submit(rt, t9);
    
    
            // Task 10: tile_add
            int32_t t10 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t10, P0, tile * 8 + 4, 0, 32, 128);
            pto_task_add_input(rt, t10, P0, tile * 8 + 5, 0, 32, 128);
            pto_task_add_input(rt, t10, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
            // Task 11: tile_add
            int32_t t11 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t11, P0, tile * 8 + 6, 0, 32, 128);
            pto_task_add_input(rt, t11, P0, tile * 8 + 7, 0, 32, 128);
            pto_task_add_input(rt, t11, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: tile_add
            int32_t t12 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t12, P1, tile * 4 + 0, 0, 32, 128);
            pto_task_add_input(rt, t12, P1, tile * 4 + 1, 0, 32, 128);
            pto_task_add_input(rt, t12, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: tile_add
            int32_t t13 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t13, P1, tile * 4 + 2, 0, 32, 128);
            pto_task_add_input(rt, t13, P1, tile * 4 + 3, 0, 32, 128);
            pto_task_add_input(rt, t13, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_add
            int32_t t14 = pto_task_alloc(rt, "tile_add", NULL, 98304, 98304, 0);
            pto_task_add_input(rt, t14, P2, tile * 2 + 0, 0, 32, 128);
            pto_task_add_input(rt, t14, P2, tile * 2 + 1, 0, 32, 128);
            pto_task_add_input(rt, t14, C, tile, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
    }

}
// =============================================================================
// Main Function for ARM64 Standalone Execution
// =============================================================================
// Usage: bgemm_dynamic [--benchmark-only] [seq_len] [tile_rows] [num_tiles] [zero]
// Flags:
//   --benchmark-only  - Only run orchestration (skip execution), output stats

int main(int argc, char** argv) {
    // Check for --benchmark-only flag
    int benchmark_only = 0;
    int arg_offset = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark-only") == 0) {
            benchmark_only = 1;
            arg_offset = 1;
            break;
        }
    }
    
    printf("============================================================\n");
    printf("  PTO ARM64 Runtime\n");
    printf("============================================================\n");
    
    // Initialize runtime (heap allocated - PTORuntime is too large for stack)
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate PTORuntime\n");
        return 1;
    }
    pto_runtime_init(rt);
    
    // Allocate test data
    float* A = (float*)calloc(1024 * 1024, sizeof(float));
    float* B = (float*)calloc(1024 * 1024, sizeof(float));
    float* C = (float*)calloc(1024 * 1024, sizeof(float));
    float* P0 = (float*)calloc(1024 * 1024, sizeof(float));
    float* P1 = (float*)calloc(1024 * 1024, sizeof(float));
    float* P2 = (float*)calloc(1024 * 1024, sizeof(float));
    int32_t seq_len = 16;  // Default, override with argv[1+arg_offset]
    int32_t tile_rows = 16;  // Default, override with argv[2+arg_offset]
    int32_t num_tiles = 16;  // Default, override with argv[3+arg_offset]
    float zero = 1.0f;  // Default test value

    // Parse command line arguments for integer parameters
    if (argc > 1 + arg_offset) seq_len = atoi(argv[1 + arg_offset]);
    if (argc > 2 + arg_offset) tile_rows = atoi(argv[2 + arg_offset]);
    if (argc > 3 + arg_offset) num_tiles = atoi(argv[3 + arg_offset]);

    
    if (benchmark_only) {
        // Benchmark mode: only measure orchestration time
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        bgemm_dynamic(rt, A, B, C, P0, P1, P2, seq_len, tile_rows, num_tiles, zero);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        long long tasks_submitted = rt->total_tasks_scheduled;
        double tasks_per_ms = tasks_submitted / time_ms;
        
        // Output in machine-parseable format
        printf("BENCHMARK: tasks=%lld time_ms=%.3f tasks_per_ms=%.2f\n",
               tasks_submitted, time_ms, tasks_per_ms);
    } else {
        // Normal execution mode
        printf("Running orchestration function: bgemm_dynamic\n");
        printf("------------------------------------------------------------\n");
        
        bgemm_dynamic(rt, A, B, C, P0, P1, P2, seq_len, tile_rows, num_tiles, zero);
        
        printf("------------------------------------------------------------\n");
        printf("Submitted %lld tasks\n", (long long)rt->total_tasks_scheduled);
        
        // Execute all tasks
        pto_execute_all(rt);
        
        printf("Execution complete!\n");
    }
    
    // Cleanup - must call shutdown before free to destroy mutexes/condvars
    fflush(stdout);
    pto_runtime_shutdown(rt);
    free(A);
    free(B);
    free(C);
    free(P0);
    free(P1);
    free(P2);
    free(rt);
    
    return 0;
}
