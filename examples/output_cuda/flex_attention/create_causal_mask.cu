// PTO Program: create_causal_mask
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: create_causal_mask
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 512 bytes (0.5 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mask                 8x8        f32       256   [  0,   2]           -
//   ones                 8x8        f32       256   [  1,  -1]           -
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

__device__ float mask[8][8];
__device__ float ones[8][8];

__global__ void create_causal_mask_kernel(float* mask_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): mask=TEXPANDS(...); ones=TEXPANDS(...); mask_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        mask[_row][_col] = -1000000000.0f;
        ones[_row][_col] = 0.0f;
        mask_mem[_row * 8 + _col] = mask[_row][_col];
    }

}

void create_causal_mask(float* mask_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    create_causal_mask_kernel<<<grid, block>>>(mask_mem);
    cudaDeviceSynchronize();
}