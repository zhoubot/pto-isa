// PTO Program: flash_attn_output_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_output_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 180,480 bytes (176.2 KB)
//   Total capacity (w/ reuse): 147,712 bytes (144.2 KB)
//   Reuse savings:            32,768 bytes (18.2%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   o_new                64x128     f32     32768   [  6,   7]           -
//   o_prev               64x128     f32     32768   [  0,   4]           -
//   o_scaled             64x128     f32     32768   [  4,   6]           -
//   p_block              64x64      f32     16384   [  1,  -1]           -
//   pv                   64x128     f32     32768   [  5,   6]           <- o_prev
//   scale_old            64x1       f32       256   [  3,   4]           -
//   v_block              64x128     f32     32768   [  2,  -1]           -
//
// BUFFER REUSE MAP:
//   pv reuses buffer of o_prev
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_output_update(float* input_o_prev, float* input_p, float* input_v, float* input_scale, float* output_o) {
    float o_prev[64][128];
    float p_block[64][64];
    float v_block[64][128];
    float scale_old[64][1];
    float o_scaled[64][128];
    float pv[64][128];
    float o_new[64][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): o_prev=TLOAD(input_o_prev,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_o_prev[_row * 128 + _col]);
            vst1q_f32(&o_prev[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_prev[_row][_col] = input_o_prev[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): p_block=TLOAD(input_p,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_p[_row * 64 + _col]);
            vst1q_f32(&p_block[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            p_block[_row][_col] = input_p[_row * 64 + _col];
        }
    }

    // FUSED LOOP (1 ops): v_block=TLOAD(input_v,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input_v[_row * 128 + _col]);
            vst1q_f32(&v_block[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            v_block[_row][_col] = input_v[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): scale_old=TLOAD(input_scale,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input_scale[_row * 1 + _col]);
            vst1q_f32(&scale_old[_row][_col], _vl3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            scale_old[_row][_col] = input_scale[_row * 1 + _col];
        }
    }

    // FUSED LOOP (1 ops): o_scaled=TROWEXPANDMUL(o_prev,scale_old)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v04 = vld1q_f32(&o_prev[_row][_col]);
            float32x4_t _vb6 = vdupq_n_f32(scale_old[_row][0]);
            float32x4_t _vr5 = vmulq_f32(_v04, _vb6);
            vst1q_f32(&o_scaled[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_scaled[_row][_col] = o_prev[_row][_col] * scale_old[_row][0];
        }
    }

    // TMATMUL: pv = p_block @ v_block
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += p_block[_i][_k] * v_block[_k][_j];}
            pv[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): o_new=TADD(o_scaled,pv); output_o=TSTORE(o_new,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v7 = vld1q_f32(&o_scaled[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&pv[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v7, _v8);
            vst1q_f32(&o_new[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&o_new[_row][_col]);
            vst1q_f32(&output_o[_row * 128 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_new[_row][_col] = o_scaled[_row][_col] + pv[_row][_col];
            output_o[_row * 128 + _col] = o_new[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "flash_attn_output_update"; }
enum { kPtoNumMemrefs = 5 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_o_prev",
    "input_p",
    "input_v",
    "input_scale",
    "output_o",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(16384),
    (size_t)(32768),
    (size_t)(256),
    (size_t)(32768),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    flash_attn_output_update((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4]);
}
#endif  // PTO_CPU_SMOKE_RUNNER