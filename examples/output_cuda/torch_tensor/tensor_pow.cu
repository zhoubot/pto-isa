// PTO Program: tensor_pow
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_pow
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            512 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_self             8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  3,   4]           <- log_self
//   scaled               8x8        f32       256   [  2,   3]           <- self
//   self                 8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   scaled reuses buffer of self
//   result reuses buffer of log_self
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
__device__ float log_self[8][8];
__device__ float scaled[8][8];
__device__ float result[8][8];

__global__ void tensor_pow_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): self=TLOAD(...); log_self=TLOG(...); scaled=TMULS(...); result=TEXP(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        log_self[_row][_col] = __logf(self[_row][_col]);
        scaled[_row][_col] = log_self[_row][_col] * 0.5f;
        result[_row][_col] = __expf(scaled[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_pow(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_pow_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}