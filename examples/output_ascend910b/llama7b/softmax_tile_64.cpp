// PTO Program: softmax_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 131,584 bytes (128.5 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            65,792 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                64x128     f32     32768   [  3,   5]           <- x
//   result               64x128     f32     32768   [  5,   6]           <- x_shifted
//   row_max              64x1       f32       256   [  1,   2]           -
//   row_sum              64x1       f32       256   [  4,   5]           <- row_max
//   x                    64x128     f32     32768   [  0,   2]           -
//   x_shifted            64x128     f32     32768   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   row_sum reuses buffer of row_max
//   result reuses buffer of x_shifted
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class softmax_tile_64Kernel {
public:
    __aicore__ inline softmax_tile_64Kernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 64);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 2 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TROWMAX: reduction max operation
        ReduceMax(row_max, x, 64);

        // FUSED (2 ops): TROWEXPANDSUB; TEXP
        BroadcastSub(x_shifted, x, row_max, 64, 8);  // row-wise broadcast subtract
        Exp(exp_x, x_shifted, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, exp_x, 64);

        // FUSED (2 ops): TROWEXPANDDIV; TSTORE
        BroadcastDiv(result, exp_x, row_sum, 64, 8);  // row-wise broadcast divide
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 64);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};

extern "C" __global__ __aicore__ void softmax_tile_64_kernel(GM_ADDR input, GM_ADDR output) {
    softmax_tile_64Kernel op;
    op.Init(input, output);
    op.Process();
}