// PTO Program: F_cross_entropy
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_cross_entropy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 1,668 bytes (1.6 KB)
//   Total capacity (w/ reuse): 836 bytes (0.8 KB)
//   Reuse savings:            832 bytes (49.9%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   ce                   8x8        f32       256   [  9,  11]           <- shifted
//   ce_row               8x1        f32        32   [ 11,  12]           <- row_sum
//   exp_shifted          8x8        f32       256   [  5,   6]           <- logits
//   log_softmax          8x8        f32       256   [  8,   9]           <- exp_shifted
//   log_sum              8x1        f32        32   [  7,   8]           -
//   logits               8x8        f32       256   [  0,   4]           -
//   result               1x1        f32         4   [ 12,  14]           -
//   row_mean             8x1        f32        32   [  2,   4]           -
//   row_sum              8x1        f32        32   [  6,   7]           <- row_mean
//   shifted              8x8        f32       256   [  4,   8]           -
//   target               8x8        f32       256   [  1,   9]           -
//
// BUFFER REUSE MAP:
//   exp_shifted reuses buffer of logits
//   row_sum reuses buffer of row_mean
//   log_softmax reuses buffer of exp_shifted
//   ce reuses buffer of shifted
//   ce_row reuses buffer of row_sum
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_cross_entropy(float* input, float* target_mem, float* output) {
    float logits[8][8];
    float target[8][8];
    float row_mean[8][1];
    float shifted[8][8];
    float exp_shifted[8][8];
    float row_sum[8][1];
    float log_sum[8][1];
    float log_softmax[8][8];
    float ce[8][8];
    float ce_row[8][1];
    float result[1][1];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (2 ops): logits=TLOAD(input,0,0); target=TLOAD(target_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&logits[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            logits[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
        }
    }

    // TROWSUM: row_mean = rowsum(logits)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += logits[_row][_col];
        }
        row_mean[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_mean=TDIVS(row_mean,8.0f)
    float32x4_t _vs2 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&row_mean[_row][_col]);
            float32x4_t _vr4 = vdivq_f32(_v3, _vs2);
            vst1q_f32(&row_mean[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
        }
    }

    // FUSED LOOP (2 ops): shifted=TROWEXPANDSUB(logits,row_mean); exp_shifted=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v05 = vld1q_f32(&logits[_row][_col]);
            float32x4_t _vb7 = vdupq_n_f32(row_mean[_row][0]);
            float32x4_t _vr6 = vsubq_f32(_v05, _vb7);
            vst1q_f32(&shifted[_row][_col], _vr6);
            float32x4_t _v8 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&exp_shifted[_row][_col], _vr9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            shifted[_row][_col] = logits[_row][_col] - row_mean[_row][0];
            exp_shifted[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_shifted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_shifted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v10 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr11 = _v10;
            vst1q_f32(&log_sum[_row][_col], _vr11);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(row_sum[_row][_col]);
        }
    }

    // FUSED LOOP (3 ops): log_softmax=TROWEXPANDSUB(shifted,log_sum); ce=TMUL(target,log_softmax); ce=TNEG(ce)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v012 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vb14 = vdupq_n_f32(log_sum[_row][0]);
            float32x4_t _vr13 = vsubq_f32(_v012, _vb14);
            vst1q_f32(&log_softmax[_row][_col], _vr13);
            float32x4_t _v15 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v16 = vld1q_f32(&log_softmax[_row][_col]);
            float32x4_t _vr17 = vmulq_f32(_v15, _v16);
            vst1q_f32(&ce[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&ce[_row][_col]);
            float32x4_t _vr19 = vnegq_f32(_v18);
            vst1q_f32(&ce[_row][_col], _vr19);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            log_softmax[_row][_col] = shifted[_row][_col] - log_sum[_row][0];
            ce[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
            ce[_row][_col] = -ce[_row][_col];
        }
    }

    // TROWSUM: ce_row = rowsum(ce)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += ce[_row][_col];
        }
        ce_row[_row][0] = _sum;}

    // TCOLSUM: result = colsum(ce_row)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += ce_row[_row][_col];
        }
        result[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(result,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs20 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v21 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr22 = vdivq_f32(_v21, _vs20);
            vst1q_f32(&result[_row][_col], _vr22);
            float32x4_t _vs23 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs23);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = result[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_cross_entropy"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "target_mem",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
    (size_t)(4),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
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
    F_cross_entropy((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER