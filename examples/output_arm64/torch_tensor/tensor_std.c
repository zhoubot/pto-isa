// PTO Program: tensor_std
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_std
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 1,104 bytes (1.1 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            296 bytes (26.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   centered             8x8        f32       256   [  5,   6]           -
//   mean_val             8x8        f32       256   [  4,   5]           -
//   result               1x1        f32         4   [ 10,  11]           <- var_total
//   row_sum              8x1        f32        32   [  1,   2]           -
//   self                 8x8        f32       256   [  0,   5]           -
//   sq_centered          8x8        f32       256   [  6,   7]           <- self
//   sq_row_sum           8x1        f32        32   [  7,   8]           <- row_sum
//   total                1x1        f32         4   [  2,   3]           -
//   var                  1x1        f32         4   [  9,  10]           -
//   var_total            1x1        f32         4   [  8,   9]           <- total
//
// BUFFER REUSE MAP:
//   sq_centered reuses buffer of self
//   sq_row_sum reuses buffer of row_sum
//   var_total reuses buffer of total
//   result reuses buffer of var_total
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_std(float* input, float* output) {
    float self[8][8];
    float row_sum[8][1];
    float total[1][1];
    float mean_val[8][8];
    float centered[8][8];
    float sq_centered[8][8];
    float sq_row_sum[8][1];
    float var_total[1][1];
    float var[1][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (1 ops): self=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWSUM: row_sum = rowsum(self)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += self[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total[0][_col] = _sum;}

    // FUSED LOOP (1 ops): total=TDIVS(total,64.0f)
    float32x4_t _vs1 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&total[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&total[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            total[_row][_col] = total[_row][_col] / 64.0f;
        }
    }

    // FUSED LOOP (3 ops): mean_val=TEXPANDS(0.0f); centered=TSUB(self,mean_val); sq_centered=TMUL(centered,centered)
    float32x4_t _vs4 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&mean_val[_row][_col], _vs4);
            float32x4_t _v5 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&mean_val[_row][_col]);
            float32x4_t _vr7 = vsubq_f32(_v5, _v6);
            vst1q_f32(&centered[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&centered[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&centered[_row][_col]);
            float32x4_t _vr10 = vmulq_f32(_v8, _v9);
            vst1q_f32(&sq_centered[_row][_col], _vr10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            mean_val[_row][_col] = 0.0f;
            centered[_row][_col] = self[_row][_col] - mean_val[_row][_col];
            sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
        }
    }

    // TROWSUM: sq_row_sum = rowsum(sq_centered)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += sq_centered[_row][_col];
        }
        sq_row_sum[_row][0] = _sum;}

    // TCOLSUM: var_total = colsum(sq_row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += sq_row_sum[_row][_col];
        }
        var_total[0][_col] = _sum;}

    // FUSED LOOP (3 ops): var=TDIVS(var_total,64.0f); result=TSQRT(var); output=TSTORE(result,0,0)
    float32x4_t _vs11 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&var_total[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs11);
            vst1q_f32(&var[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&var[_row][_col]);
            float32x4_t _vr15 = vsqrtq_f32(_v14);
            vst1q_f32(&result[_row][_col], _vr15);
            float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs16);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            var[_row][_col] = var_total[_row][_col] / 64.0f;
            result[_row][_col] = sqrtf(var[_row][_col]);
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_std"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(4),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    tensor_std((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER