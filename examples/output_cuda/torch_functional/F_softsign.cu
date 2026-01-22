// PTO Program: F_softsign
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_softsign
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            256 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_x                8x8        f32       256   [  1,   2]           -
//   one_plus_abs         8x8        f32       256   [  2,   3]           -
//   result               8x8        f32       256   [  3,   4]           <- abs_x
//   x                    8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of abs_x
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
__device__ float abs_x[8][8];
__device__ float one_plus_abs[8][8];
__device__ float result[8][8];

__global__ void F_softsign_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): x=TLOAD(...); abs_x=TABS(...); one_plus_abs=TADDS(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        abs_x[_row][_col] = fabsf(x[_row][_col]);
        one_plus_abs[_row][_col] = abs_x[_row][_col] + 1.0f;
        result[_row][_col] = x[_row][_col] / one_plus_abs[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_softsign(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_softsign_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}