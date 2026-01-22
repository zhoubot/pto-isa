// PTO Program: F_hardswish
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_hardswish
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 2,048 bytes (2.0 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            768 bytes (37.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   clamp_low            8x8        f32       256   [  5,   6]           -
//   hardsig              8x8        f32       256   [  6,   7]           <- scaled
//   ones                 8x8        f32       256   [  4,   6]           -
//   result               8x8        f32       256   [  7,   8]           <- zeros
//   scaled               8x8        f32       256   [  2,   5]           -
//   x                    8x8        f32       256   [  0,   7]           -
//   x_plus_3             8x8        f32       256   [  1,   2]           -
//   zeros                8x8        f32       256   [  3,   5]           <- x_plus_3
//
// BUFFER REUSE MAP:
//   zeros reuses buffer of x_plus_3
//   hardsig reuses buffer of scaled
//   result reuses buffer of zeros
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

__device__ float x[8][8];
__device__ float x_plus_3[8][8];
__device__ float scaled[8][8];
__device__ float zeros[8][8];
__device__ float ones[8][8];
__device__ float clamp_low[8][8];
__device__ float hardsig[8][8];
__device__ float result[8][8];

__global__ void F_hardswish_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (9 ops): x=TLOAD(...); x_plus_3=TADDS(...); scaled=TDIVS(...); zeros=TEXPANDS(...); ones=TEXPANDS(...); clamp_low=TMAX(...); hardsig=TMIN(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
        scaled[_row][_col] = x_plus_3[_row][_col] / 6.0f;
        zeros[_row][_col] = 0.0f;
        ones[_row][_col] = 1.0f;
        clamp_low[_row][_col] = fmaxf(scaled[_row][_col], zeros[_row][_col]);
        hardsig[_row][_col] = fminf(clamp_low[_row][_col], ones[_row][_col]);
        result[_row][_col] = x[_row][_col] * hardsig[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_hardswish(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_hardswish_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}