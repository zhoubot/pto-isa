// PTO Program: llama_layer_dynamic
// Function Type: Orchestration (control flow only)
// Orchestration function - builds task graph using PTO runtime
#include "pto_runtime.h"
#include "pto_runtime.c"  // Include for standalone build

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void llama_layer_dynamic(PTORuntime* rt, float* input, float* output, float* attn_norm_weights, float* wq, float* wk, float* wv, float* wo, float* cos_cache, float* sin_cache, float* mlp_norm_weights, float* w_gate, float* w_up, float* w_down, float* all_q_tiles, float* all_k_tiles, float* all_v_tiles, float* all_q_rope, float* all_k_rope, float* all_attn_out, float* all_m_vec, float* all_l_vec, float* all_hidden, float* temp_norm, float* temp_scores, float* temp_attn_weights, float* temp_scale, float* temp_gate, float* temp_up, float* temp_swiglu, float* temp_mlp_out, float* const_zeros_large, float* const_zeros_small, float* const_neg_inf, int32_t seq_len, int32_t num_tiles) {

    // Loop fusion: 0 loop overheads saved

    int tile_rows = 32;

    int zero = 0;

    // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
    // Larger tiles → fewer iterations → smaller task graph!
    {
        int _tile_i_base = 0;
        int _tile_i_limit = num_tiles;
        int _tile_i_base_rows = 0;  // Track rows processed in base tile units

        int _tile_i_residual = _tile_i_limit & 255;  // Residual base-tiles < 256
        int _tile_i_quantized = _tile_i_limit - _tile_i_residual;  // Quantized to 256 boundary

        // Power-of-2 blocks with adaptive iteration counts:
        // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
        if (_tile_i_quantized & 2048) {
            for (int tile_i = 0; tile_i < 1024; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t0, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t0, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t1, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
            pto_task_add_output(rt, t1, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t2, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
            pto_task_add_output(rt, t2, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t3, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
            pto_task_add_output(rt, t3, all_v_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t4, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t4, all_q_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t5, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t5, all_k_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t5);
    
    
            }
            _tile_i_base_rows += 2048;  // Advance by 2048 base-tiles
        }
        // Block 1024: tile_rows=64, scale=2x, actual_iters=512
        if (_tile_i_quantized & 1024) {
            for (int tile_i = 0; tile_i < 512; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t0, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t0, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t1, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
            pto_task_add_output(rt, t1, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t2, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
            pto_task_add_output(rt, t2, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t3, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
            pto_task_add_output(rt, t3, all_v_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t4, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t4, all_q_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t5, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t5, all_k_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t5);
    
    
            }
            _tile_i_base_rows += 1024;  // Advance by 1024 base-tiles
        }
        // Block 512: tile_rows=64, scale=2x, actual_iters=256
        if (_tile_i_quantized & 512) {
            for (int tile_i = 0; tile_i < 256; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t0, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t0, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t1, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
            pto_task_add_output(rt, t1, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t2, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
            pto_task_add_output(rt, t2, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t3, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
            pto_task_add_output(rt, t3, all_v_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t4, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t4, all_q_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t5, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t5, all_k_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t5);
    
    
            }
            _tile_i_base_rows += 512;  // Advance by 512 base-tiles
        }
        // Block 256: tile_rows=64, scale=2x, actual_iters=128
        if (_tile_i_quantized & 256) {
            for (int tile_i = 0; tile_i < 128; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t0, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t0, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t1, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
            pto_task_add_output(rt, t1, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t2, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
            pto_task_add_output(rt, t2, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t3, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
            pto_task_add_output(rt, t3, all_v_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t4, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t4, all_q_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t5, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t5, all_k_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t5);
    
    
            }
            _tile_i_base_rows += 256;  // Advance by 256 base-tiles
        }

        // Residual loop for remaining iterations < 256 (tile_rows=64)
        if (_tile_i_residual > 0) {
            for (int tile_i = 0; tile_i < _tile_i_residual; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + tile_i;  // Offset in base-tile units
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t0, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t0, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t1, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
            pto_task_add_output(rt, t1, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t2, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
            pto_task_add_output(rt, t2, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t3, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
            pto_task_add_output(rt, t3, all_v_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t4, all_q_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t4, all_q_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile_64", NULL, 100663296, 67108864);
            pto_task_add_input(rt, t5, all_k_tiles, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
            pto_task_add_output(rt, t5, all_k_rope, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t5);
    
    
            }
        }
    }

    // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
    // Larger tiles → fewer iterations → smaller task graph!
    {
        int _q_tile_base = 0;
        int _q_tile_limit = num_tiles;
        int _q_tile_base_rows = 0;  // Track rows processed in base tile units

        int _q_tile_residual = _q_tile_limit & 255;  // Residual base-tiles < 256
        int _q_tile_quantized = _q_tile_limit - _q_tile_residual;  // Quantized to 256 boundary

        // Power-of-2 blocks with adaptive iteration counts:
        // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
        if (_q_tile_quantized & 2048) {
            for (int q_tile = 0; q_tile < 1024; q_tile += 1) {
                int _q_tile_row_offset = _q_tile_base_rows + (q_tile * 2);  // Offset in base-tile units
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 34078720, 34078720);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 8, 8);
            pto_task_add_output(rt, t6, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_m_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t6);
    
    
            // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
            // Larger tiles → fewer iterations → smaller task graph!
            {
                int _kv_tile_base = 0;
                int _kv_tile_limit = num_tiles;
                int _kv_tile_base_rows = 0;  // Track rows processed in base tile units

                int _kv_tile_residual = _kv_tile_limit & 255;  // Residual base-tiles < 256
                int _kv_tile_quantized = _kv_tile_limit - _kv_tile_residual;  // Quantized to 256 boundary

                // Power-of-2 blocks with adaptive iteration counts:
                // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
                if (_kv_tile_quantized & 2048) {
                    for (int kv_tile = 0; kv_tile < 1024; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 2048;  // Advance by 2048 base-tiles
                }
                // Block 1024: tile_rows=64, scale=2x, actual_iters=512
                if (_kv_tile_quantized & 1024) {
                    for (int kv_tile = 0; kv_tile < 512; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 1024;  // Advance by 1024 base-tiles
                }
                // Block 512: tile_rows=64, scale=2x, actual_iters=256
                if (_kv_tile_quantized & 512) {
                    for (int kv_tile = 0; kv_tile < 256; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 512;  // Advance by 512 base-tiles
                }
                // Block 256: tile_rows=64, scale=2x, actual_iters=128
                if (_kv_tile_quantized & 256) {
                    for (int kv_tile = 0; kv_tile < 128; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 256;  // Advance by 256 base-tiles
                }

                // Residual loop for remaining iterations < 256 (tile_rows=64)
                if (_kv_tile_residual > 0) {
                    for (int kv_tile = 0; kv_tile < _kv_tile_residual; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + kv_tile;  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                }
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 67371008, 67371008);
            pto_task_add_input(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t10, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t10);
    
    
            }
            _q_tile_base_rows += 2048;  // Advance by 2048 base-tiles
        }
        // Block 1024: tile_rows=64, scale=2x, actual_iters=512
        if (_q_tile_quantized & 1024) {
            for (int q_tile = 0; q_tile < 512; q_tile += 1) {
                int _q_tile_row_offset = _q_tile_base_rows + (q_tile * 2);  // Offset in base-tile units
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 34078720, 34078720);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 8, 8);
            pto_task_add_output(rt, t6, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_m_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t6);
    
    
            // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
            // Larger tiles → fewer iterations → smaller task graph!
            {
                int _kv_tile_base = 0;
                int _kv_tile_limit = num_tiles;
                int _kv_tile_base_rows = 0;  // Track rows processed in base tile units

                int _kv_tile_residual = _kv_tile_limit & 255;  // Residual base-tiles < 256
                int _kv_tile_quantized = _kv_tile_limit - _kv_tile_residual;  // Quantized to 256 boundary

                // Power-of-2 blocks with adaptive iteration counts:
                // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
                if (_kv_tile_quantized & 2048) {
                    for (int kv_tile = 0; kv_tile < 1024; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 2048;  // Advance by 2048 base-tiles
                }
                // Block 1024: tile_rows=64, scale=2x, actual_iters=512
                if (_kv_tile_quantized & 1024) {
                    for (int kv_tile = 0; kv_tile < 512; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 1024;  // Advance by 1024 base-tiles
                }
                // Block 512: tile_rows=64, scale=2x, actual_iters=256
                if (_kv_tile_quantized & 512) {
                    for (int kv_tile = 0; kv_tile < 256; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 512;  // Advance by 512 base-tiles
                }
                // Block 256: tile_rows=64, scale=2x, actual_iters=128
                if (_kv_tile_quantized & 256) {
                    for (int kv_tile = 0; kv_tile < 128; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 256;  // Advance by 256 base-tiles
                }

                // Residual loop for remaining iterations < 256 (tile_rows=64)
                if (_kv_tile_residual > 0) {
                    for (int kv_tile = 0; kv_tile < _kv_tile_residual; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + kv_tile;  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                }
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 67371008, 67371008);
            pto_task_add_input(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t10, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t10);
    
    
            }
            _q_tile_base_rows += 1024;  // Advance by 1024 base-tiles
        }
        // Block 512: tile_rows=64, scale=2x, actual_iters=256
        if (_q_tile_quantized & 512) {
            for (int q_tile = 0; q_tile < 256; q_tile += 1) {
                int _q_tile_row_offset = _q_tile_base_rows + (q_tile * 2);  // Offset in base-tile units
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 34078720, 34078720);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 8, 8);
            pto_task_add_output(rt, t6, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_m_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t6);
    
    
            // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
            // Larger tiles → fewer iterations → smaller task graph!
            {
                int _kv_tile_base = 0;
                int _kv_tile_limit = num_tiles;
                int _kv_tile_base_rows = 0;  // Track rows processed in base tile units

                int _kv_tile_residual = _kv_tile_limit & 255;  // Residual base-tiles < 256
                int _kv_tile_quantized = _kv_tile_limit - _kv_tile_residual;  // Quantized to 256 boundary

                // Power-of-2 blocks with adaptive iteration counts:
                // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
                if (_kv_tile_quantized & 2048) {
                    for (int kv_tile = 0; kv_tile < 1024; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 2048;  // Advance by 2048 base-tiles
                }
                // Block 1024: tile_rows=64, scale=2x, actual_iters=512
                if (_kv_tile_quantized & 1024) {
                    for (int kv_tile = 0; kv_tile < 512; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 1024;  // Advance by 1024 base-tiles
                }
                // Block 512: tile_rows=64, scale=2x, actual_iters=256
                if (_kv_tile_quantized & 512) {
                    for (int kv_tile = 0; kv_tile < 256; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 512;  // Advance by 512 base-tiles
                }
                // Block 256: tile_rows=64, scale=2x, actual_iters=128
                if (_kv_tile_quantized & 256) {
                    for (int kv_tile = 0; kv_tile < 128; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 256;  // Advance by 256 base-tiles
                }

                // Residual loop for remaining iterations < 256 (tile_rows=64)
                if (_kv_tile_residual > 0) {
                    for (int kv_tile = 0; kv_tile < _kv_tile_residual; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + kv_tile;  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                }
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 67371008, 67371008);
            pto_task_add_input(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t10, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t10);
    
    
            }
            _q_tile_base_rows += 512;  // Advance by 512 base-tiles
        }
        // Block 256: tile_rows=64, scale=2x, actual_iters=128
        if (_q_tile_quantized & 256) {
            for (int q_tile = 0; q_tile < 128; q_tile += 1) {
                int _q_tile_row_offset = _q_tile_base_rows + (q_tile * 2);  // Offset in base-tile units
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 34078720, 34078720);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 8, 8);
            pto_task_add_output(rt, t6, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_m_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t6);
    
    
            // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
            // Larger tiles → fewer iterations → smaller task graph!
            {
                int _kv_tile_base = 0;
                int _kv_tile_limit = num_tiles;
                int _kv_tile_base_rows = 0;  // Track rows processed in base tile units

                int _kv_tile_residual = _kv_tile_limit & 255;  // Residual base-tiles < 256
                int _kv_tile_quantized = _kv_tile_limit - _kv_tile_residual;  // Quantized to 256 boundary

                // Power-of-2 blocks with adaptive iteration counts:
                // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
                if (_kv_tile_quantized & 2048) {
                    for (int kv_tile = 0; kv_tile < 1024; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 2048;  // Advance by 2048 base-tiles
                }
                // Block 1024: tile_rows=64, scale=2x, actual_iters=512
                if (_kv_tile_quantized & 1024) {
                    for (int kv_tile = 0; kv_tile < 512; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 1024;  // Advance by 1024 base-tiles
                }
                // Block 512: tile_rows=64, scale=2x, actual_iters=256
                if (_kv_tile_quantized & 512) {
                    for (int kv_tile = 0; kv_tile < 256; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 512;  // Advance by 512 base-tiles
                }
                // Block 256: tile_rows=64, scale=2x, actual_iters=128
                if (_kv_tile_quantized & 256) {
                    for (int kv_tile = 0; kv_tile < 128; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 256;  // Advance by 256 base-tiles
                }

                // Residual loop for remaining iterations < 256 (tile_rows=64)
                if (_kv_tile_residual > 0) {
                    for (int kv_tile = 0; kv_tile < _kv_tile_residual; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + kv_tile;  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                }
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 67371008, 67371008);
            pto_task_add_input(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t10, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t10);
    
    
            }
            _q_tile_base_rows += 256;  // Advance by 256 base-tiles
        }

        // Residual loop for remaining iterations < 256 (tile_rows=64)
        if (_q_tile_residual > 0) {
            for (int q_tile = 0; q_tile < _q_tile_residual; q_tile += 1) {
                int _q_tile_row_offset = _q_tile_base_rows + q_tile;  // Offset in base-tile units
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 34078720, 34078720);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 8, 8);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 8, 8);
            pto_task_add_output(rt, t6, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t6, all_m_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t6);
    
    
            // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
            // Larger tiles → fewer iterations → smaller task graph!
            {
                int _kv_tile_base = 0;
                int _kv_tile_limit = num_tiles;
                int _kv_tile_base_rows = 0;  // Track rows processed in base tile units

                int _kv_tile_residual = _kv_tile_limit & 255;  // Residual base-tiles < 256
                int _kv_tile_quantized = _kv_tile_limit - _kv_tile_residual;  // Quantized to 256 boundary

                // Power-of-2 blocks with adaptive iteration counts:
                // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
                if (_kv_tile_quantized & 2048) {
                    for (int kv_tile = 0; kv_tile < 1024; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 2048;  // Advance by 2048 base-tiles
                }
                // Block 1024: tile_rows=64, scale=2x, actual_iters=512
                if (_kv_tile_quantized & 1024) {
                    for (int kv_tile = 0; kv_tile < 512; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 1024;  // Advance by 1024 base-tiles
                }
                // Block 512: tile_rows=64, scale=2x, actual_iters=256
                if (_kv_tile_quantized & 512) {
                    for (int kv_tile = 0; kv_tile < 256; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 512;  // Advance by 512 base-tiles
                }
                // Block 256: tile_rows=64, scale=2x, actual_iters=128
                if (_kv_tile_quantized & 256) {
                    for (int kv_tile = 0; kv_tile < 128; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + (kv_tile * 2);  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                    _kv_tile_base_rows += 256;  // Advance by 256 base-tiles
                }

                // Residual loop for remaining iterations < 256 (tile_rows=64)
                if (_kv_tile_residual > 0) {
                    for (int kv_tile = 0; kv_tile < _kv_tile_residual; kv_tile += 1) {
                        int _kv_tile_row_offset = _kv_tile_base_rows + kv_tile;  // Offset in base-tile units
        
                    // Task 7: flash_attn_score_block
                    int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 100663296, 100663296);
                    pto_task_add_input(rt, t7, all_q_rope, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t7, all_k_rope, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t7, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t7);
        
        
                    // Task 8: flash_attn_softmax_update
                    int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 52690944, 34865152);
                    pto_task_add_input(rt, t8, temp_scores, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_m_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, all_l_vec, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t8, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t8);
        
        
                    // Task 9: flash_attn_output_update
                    int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 184811520, 151257088);
                    pto_task_add_input(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_attn_weights, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, all_v_tiles, _kv_tile_row_offset, 0, 8, 8);
                    pto_task_add_input(rt, t9, temp_scale, _q_tile_row_offset, 0, 8, 8);
                    pto_task_add_output(rt, t9, all_attn_out, _q_tile_row_offset, 0, 8, 8);
                    pto_task_submit(rt, t9);
        
        
                    }
                }
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 67371008, 67371008);
            pto_task_add_input(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t10, all_l_vec, _q_tile_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t10, all_attn_out, _q_tile_row_offset, 0, 8, 8);
            pto_task_submit(rt, t10);
    
    
            }
        }
    }

    // Binary-expanded loop with ADAPTIVE tile sizes: num_tiles (tile_levels: {4096: 64, 2048: 64, 1024: 64, 512: 64, 256: 64})
    // Larger tiles → fewer iterations → smaller task graph!
    {
        int _tile_i_base = 0;
        int _tile_i_limit = num_tiles;
        int _tile_i_base_rows = 0;  // Track rows processed in base tile units

        int _tile_i_residual = _tile_i_limit & 255;  // Residual base-tiles < 256
        int _tile_i_quantized = _tile_i_limit - _tile_i_residual;  // Quantized to 256 boundary

        // Power-of-2 blocks with adaptive iteration counts:
        // Block 2048: tile_rows=64, scale=2x, actual_iters=1024
        if (_tile_i_quantized & 2048) {
            for (int tile_i = 0; tile_i < 1024; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t11, all_attn_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t11, wo, 0, 0, 8, 8);
            pto_task_add_output(rt, t11, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t12, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t12, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t12, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t13, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t13, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t14, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 8, 8);
            pto_task_add_output(rt, t14, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t15, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t15, w_up, 0, 0, 8, 8);
            pto_task_add_output(rt, t15, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile_64", NULL, 134217728, 67108864);
            pto_task_add_input(rt, t16, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t16, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t16, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t17, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t17, w_down, 0, 0, 8, 8);
            pto_task_add_output(rt, t17, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t18, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t18, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t18, output, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t18);
    
    
            }
            _tile_i_base_rows += 2048;  // Advance by 2048 base-tiles
        }
        // Block 1024: tile_rows=64, scale=2x, actual_iters=512
        if (_tile_i_quantized & 1024) {
            for (int tile_i = 0; tile_i < 512; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t11, all_attn_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t11, wo, 0, 0, 8, 8);
            pto_task_add_output(rt, t11, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t12, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t12, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t12, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t13, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t13, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t14, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 8, 8);
            pto_task_add_output(rt, t14, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t15, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t15, w_up, 0, 0, 8, 8);
            pto_task_add_output(rt, t15, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile_64", NULL, 134217728, 67108864);
            pto_task_add_input(rt, t16, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t16, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t16, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t17, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t17, w_down, 0, 0, 8, 8);
            pto_task_add_output(rt, t17, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t18, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t18, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t18, output, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t18);
    
    
            }
            _tile_i_base_rows += 1024;  // Advance by 1024 base-tiles
        }
        // Block 512: tile_rows=64, scale=2x, actual_iters=256
        if (_tile_i_quantized & 512) {
            for (int tile_i = 0; tile_i < 256; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t11, all_attn_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t11, wo, 0, 0, 8, 8);
            pto_task_add_output(rt, t11, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t12, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t12, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t12, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t13, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t13, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t14, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 8, 8);
            pto_task_add_output(rt, t14, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t15, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t15, w_up, 0, 0, 8, 8);
            pto_task_add_output(rt, t15, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile_64", NULL, 134217728, 67108864);
            pto_task_add_input(rt, t16, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t16, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t16, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t17, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t17, w_down, 0, 0, 8, 8);
            pto_task_add_output(rt, t17, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t18, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t18, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t18, output, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t18);
    
    
            }
            _tile_i_base_rows += 512;  // Advance by 512 base-tiles
        }
        // Block 256: tile_rows=64, scale=2x, actual_iters=128
        if (_tile_i_quantized & 256) {
            for (int tile_i = 0; tile_i < 128; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + (tile_i * 2);  // Offset in base-tile units
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t11, all_attn_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t11, wo, 0, 0, 8, 8);
            pto_task_add_output(rt, t11, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t12, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t12, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t12, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t13, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t13, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t14, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 8, 8);
            pto_task_add_output(rt, t14, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t15, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t15, w_up, 0, 0, 8, 8);
            pto_task_add_output(rt, t15, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile_64", NULL, 134217728, 67108864);
            pto_task_add_input(rt, t16, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t16, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t16, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t17, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t17, w_down, 0, 0, 8, 8);
            pto_task_add_output(rt, t17, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t18, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t18, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t18, output, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t18);
    
    
            }
            _tile_i_base_rows += 256;  // Advance by 256 base-tiles
        }

        // Residual loop for remaining iterations < 256 (tile_rows=64)
        if (_tile_i_residual > 0) {
            for (int tile_i = 0; tile_i < _tile_i_residual; tile_i += 1) {
                int _tile_i_row_offset = _tile_i_base_rows + tile_i;  // Offset in base-tile units
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t11, all_attn_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t11, wo, 0, 0, 8, 8);
            pto_task_add_output(rt, t11, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t12, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t12, input, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t12, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile_64", NULL, 84279296, 50593792);
            pto_task_add_input(rt, t13, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 8, 8);
            pto_task_add_output(rt, t13, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t14, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 8, 8);
            pto_task_add_output(rt, t14, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t15, temp_norm, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t15, w_up, 0, 0, 8, 8);
            pto_task_add_output(rt, t15, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile_64", NULL, 134217728, 67108864);
            pto_task_add_input(rt, t16, temp_gate, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t16, temp_up, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t16, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul_64", NULL, 100663296, 100663296);
            pto_task_add_input(rt, t17, temp_swiglu, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t17, w_down, 0, 0, 8, 8);
            pto_task_add_output(rt, t17, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile_64", NULL, 50331648, 50331648);
            pto_task_add_input(rt, t18, temp_mlp_out, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_input(rt, t18, all_hidden, _tile_i_row_offset, 0, 8, 8);
            pto_task_add_output(rt, t18, output, _tile_i_row_offset, 0, 8, 8);
            pto_task_submit(rt, t18);
    
    
            }
        }
    }

}

/**
 * Main: Build task graph and dump to file
 */
int main(int argc, char** argv) {
    // Allocate runtime on heap (PTORuntime is ~187MB with 65536 max tasks)
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) { fprintf(stderr, "Failed to allocate PTORuntime\n"); return 1; }
    pto_runtime_init(rt);

    // Declare dummy buffers
    float input[1024];  // Dummy buffer
    float output[1024];  // Dummy buffer
    float attn_norm_weights[1024];  // Dummy buffer
    float wq[1024];  // Dummy buffer
    float wk[1024];  // Dummy buffer
    float wv[1024];  // Dummy buffer
    float wo[1024];  // Dummy buffer
    float cos_cache[1024];  // Dummy buffer
    float sin_cache[1024];  // Dummy buffer
    float mlp_norm_weights[1024];  // Dummy buffer
    float w_gate[1024];  // Dummy buffer
    float w_up[1024];  // Dummy buffer
    float w_down[1024];  // Dummy buffer
    float all_q_tiles[1024];  // Dummy buffer
    float all_k_tiles[1024];  // Dummy buffer
    float all_v_tiles[1024];  // Dummy buffer
    float all_q_rope[1024];  // Dummy buffer
    float all_k_rope[1024];  // Dummy buffer
    float all_attn_out[1024];  // Dummy buffer
    float all_m_vec[1024];  // Dummy buffer
    float all_l_vec[1024];  // Dummy buffer
    float all_hidden[1024];  // Dummy buffer
    float temp_norm[1024];  // Dummy buffer
    float temp_scores[1024];  // Dummy buffer
    float temp_attn_weights[1024];  // Dummy buffer
    float temp_scale[1024];  // Dummy buffer
    float temp_gate[1024];  // Dummy buffer
    float temp_up[1024];  // Dummy buffer
    float temp_swiglu[1024];  // Dummy buffer
    float temp_mlp_out[1024];  // Dummy buffer
    float const_zeros_large[1024];  // Dummy buffer
    float const_zeros_small[1024];  // Dummy buffer
    float const_neg_inf[1024];  // Dummy buffer

    int seq_len = 16384;  // TODO: set from args
    int num_tiles = 256;  // TODO: set from args

    // Build task graph
    llama_layer_dynamic(rt, input, output, attn_norm_weights, wq, wk, wv, wo, cos_cache, sin_cache, mlp_norm_weights, w_gate, w_up, w_down, all_q_tiles, all_k_tiles, all_v_tiles, all_q_rope, all_k_rope, all_attn_out, all_m_vec, all_l_vec, all_hidden, temp_norm, temp_scores, temp_attn_weights, temp_scale, temp_gate, temp_up, temp_swiglu, temp_mlp_out, const_zeros_large, const_zeros_small, const_neg_inf, seq_len, num_tiles);

    printf("\n");
    pto_runtime_dump_stdout(rt);
    pto_runtime_dump(rt, "llama_layer_dynamic_task_graph.txt");

    pto_runtime_shutdown(rt);
    free(rt);
    return 0;
}