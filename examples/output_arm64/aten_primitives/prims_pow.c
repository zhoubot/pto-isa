// PTO Program: prims_pow
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: prims_pow
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 81,920 bytes (80.0 KB)
//   Total capacity (w/ reuse): 81,920 bytes (80.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   base                 1x4096     f32     16384   [  3,  14]           -
//   exp_tile             1x4096     f32     16384   [  4,  15]           -
//   log_base             1x4096     f32     16384   [  5,  15]           -
//   product              1x4096     f32     16384   [  6,  16]           -
//   result               1x4096     f32     16384   [  7,  17]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void prims_pow(float* input_base, float* input_exp, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    float base[1][4096];
    float exp_tile[1][4096];
    float log_base[1][4096];
    float product[1][4096];
    float result[1][4096];

    // Loop fusion: 10 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (6 ops): base=TLOAD(input_base,tile_idx,0); exp_tile=TLOAD(input_exp,tile_idx,0); log_base=TLOG(base); product=TMUL(exp_tile,log_base); result=TEXP(product); output=TSTORE(result,tile_idx,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl0 = vld1q_f32(&input_base[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&base[_row][_col], _vl0);
                float32x4_t _vl1 = vld1q_f32(&input_exp[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&exp_tile[_row][_col], _vl1);
                float32x4_t _v2 = vld1q_f32(&base[_row][_col]);
                float32x4_t _vr3 = _v2;
                vst1q_f32(&log_base[_row][_col], _vr3);
                float32x4_t _v4 = vld1q_f32(&exp_tile[_row][_col]);
                float32x4_t _v5 = vld1q_f32(&log_base[_row][_col]);
                float32x4_t _vr6 = vmulq_f32(_v4, _v5);
                vst1q_f32(&product[_row][_col], _vr6);
                float32x4_t _v7 = vld1q_f32(&product[_row][_col]);
                float32x4_t _vr8 = _v7;
                vst1q_f32(&result[_row][_col], _vr8);
                float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs9);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                base[_row][_col] = input_base[(tile_idx) * 4096 + _row * 4096 + _col];
                exp_tile[_row][_col] = input_exp[(tile_idx) * 4096 + _row * 4096 + _col];
                log_base[_row][_col] = logf(base[_row][_col]);
                product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
                result[_row][_col] = expf(product[_row][_col]);
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (6 ops): base=TLOAD(input_base,num_full_tiles,0); exp_tile=TLOAD(input_exp,num_full_tiles,0); log_base=TLOG(base); product=TMUL(exp_tile,log_base); result=TEXP(product); output=TSTORE(result,num_full_tiles,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl10 = vld1q_f32(&input_base[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&base[_row][_col], _vl10);
                float32x4_t _vl11 = vld1q_f32(&input_exp[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&exp_tile[_row][_col], _vl11);
                float32x4_t _v12 = vld1q_f32(&base[_row][_col]);
                float32x4_t _vr13 = _v12;
                vst1q_f32(&log_base[_row][_col], _vr13);
                float32x4_t _v14 = vld1q_f32(&exp_tile[_row][_col]);
                float32x4_t _v15 = vld1q_f32(&log_base[_row][_col]);
                float32x4_t _vr16 = vmulq_f32(_v14, _v15);
                vst1q_f32(&product[_row][_col], _vr16);
                float32x4_t _v17 = vld1q_f32(&product[_row][_col]);
                float32x4_t _vr18 = _v17;
                vst1q_f32(&result[_row][_col], _vr18);
                float32x4_t _vs19 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs19);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                base[_row][_col] = input_base[(num_full_tiles) * 4096 + _row * 4096 + _col];
                exp_tile[_row][_col] = input_exp[(num_full_tiles) * 4096 + _row * 4096 + _col];
                log_base[_row][_col] = logf(base[_row][_col]);
                product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
                result[_row][_col] = expf(product[_row][_col]);
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "prims_pow"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_base",
    "input_exp",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
    (size_t)(16384),
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
    prims_pow((float*)args[0], (float*)args[1], (float*)args[2], (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER