// PTO Program: tensor_lerp
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_lerp
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            512 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  2,   3]           -
//   end                  8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  4,   5]           <- diff
//   scaled               8x8        f32       256   [  3,   4]           <- end
//   self                 8x8        f32       256   [  0,   4]           -
//
// BUFFER REUSE MAP:
//   scaled reuses buffer of end
//   result reuses buffer of diff
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
__device__ float end[8][8];
__device__ float diff[8][8];
__device__ float scaled[8][8];
__device__ float result[8][8];

__global__ void tensor_lerp_kernel(float* input_self, float* input_end, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): self=TLOAD(...); end=TLOAD(...); diff=TSUB(...); scaled=TMULS(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        end[_row][_col] = input_end[_row * 8 + _col];
        diff[_row][_col] = end[_row][_col] - self;
        scaled[_row][_col] = diff[_row][_col] * 0.5f;
        result[_row][_col] = self[_row][_col] + scaled[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_lerp(float* input_self, float* input_end, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_lerp_kernel<<<grid, block>>>(input_self, input_end, output);
    cudaDeviceSynchronize();
}