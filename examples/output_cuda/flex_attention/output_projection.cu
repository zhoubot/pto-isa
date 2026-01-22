// PTO Program: output_projection
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: output_projection
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 4,352 bytes (4.2 KB)
//   Total capacity (w/ reuse): 4,352 bytes (4.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   W_O                  8x64       f32      2048   [  1,  -1]           -
//   attn_out             8x8        f32       256   [  0,  -1]           -
//   output               8x64       f32      2048   [  2,   3]           -
//
// ======================================================================

// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ float attn_out[8][8];
__device__ float W_O[8][64];
__device__ float output[8][64];

__global__ void output_projection_kernel(float* attn_mem, float* WO_mem, float* output_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): attn_out=TLOAD(...)
    if (_row < 8 && _col < 8) {
        attn_out[_row][_col] = attn_mem[_row * 8 + _col];
    }

    // FUSED (1 ops): W_O=TLOAD(...)
    if (_row < 8 && _col < 64) {
        W_O[_row][_col] = WO_mem[_row * 64 + _col];
    }

    // TMATMUL: output = attn_out @ W_O
    if (_row < 8 && _col < 64) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += attn_out[_row][_k] * W_O[_k][_col];
        output[_row][_col] = _sum;}

    // FUSED (1 ops): output_mem=TSTORE(...)
    if (_row < 8 && _col < 64) {
        output_mem[_row * 64 + _col] = output[_row][_col];
    }

}

void output_projection(float* attn_mem, float* WO_mem, float* output_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    output_projection_kernel<<<grid, block>>>(attn_mem, WO_mem, output_mem);
    cudaDeviceSynchronize();
}