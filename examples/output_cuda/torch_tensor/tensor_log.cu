// PTO Program: tensor_log
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_log
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
//   result               8x8        f32       256   [  1,   2]           -
//   self                 8x8        f32       256   [  0,   1]           -
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

__device__ float self[8][8];
__device__ float result[8][8];

__global__ void tensor_log_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): self=TLOAD(...); result=TLOG(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = __logf(self[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_log(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_log_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}