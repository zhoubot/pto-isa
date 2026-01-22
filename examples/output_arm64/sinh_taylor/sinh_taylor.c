// PTO Program: sinh_taylor
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: sinh_taylor
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 65,536 bytes (64.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               32x128     f32     16384   [  4,  51]           -
//   term                 32x128     f32     16384   [  6,  50]           -
//   x                    32x128     f32     16384   [  3,  32]           -
//   x_squared            32x128     f32     16384   [  5,  48]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void sinh_taylor(float* input, float* output, int32_t total_elements, int32_t num_full_tiles, int32_t tail_elements, int32_t offset) {
    float x[32][128];
    float x_squared[32][128];
    float term[32][128];
    float result[32][128];

    // Loop fusion: 44 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (23 ops): x=TLOAD(input,tile_idx,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,110.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,156.0f); result=TADD(result,term); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(1.0f);
        float32x4_t _vs1 = vdupq_n_f32(6.0f);
        float32x4_t _vs2 = vdupq_n_f32(20.0f);
        float32x4_t _vs3 = vdupq_n_f32(42.0f);
        float32x4_t _vs4 = vdupq_n_f32(72.0f);
        float32x4_t _vs5 = vdupq_n_f32(110.0f);
        float32x4_t _vs6 = vdupq_n_f32(156.0f);
        for (int _row = 0; _row < 32; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 128; _col += 4) {
                float32x4_t _vl7 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 128 + _col]);
                vst1q_f32(&x[_row][_col], _vl7);
                float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr9 = vmulq_f32(_v8, _vs0);
                vst1q_f32(&result[_row][_col], _vr9);
                float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v11 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr12 = vmulq_f32(_v10, _v11);
                vst1q_f32(&x_squared[_row][_col], _vr12);
                float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr14 = vmulq_f32(_v13, _vs0);
                vst1q_f32(&term[_row][_col], _vr14);
                float32x4_t _v15 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v16 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr17 = vmulq_f32(_v15, _v16);
                vst1q_f32(&term[_row][_col], _vr17);
                float32x4_t _v18 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr19 = vdivq_f32(_v18, _vs1);
                vst1q_f32(&term[_row][_col], _vr19);
                float32x4_t _v20 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v21 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr22 = vaddq_f32(_v20, _v21);
                vst1q_f32(&result[_row][_col], _vr22);
                float32x4_t _v23 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v24 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr25 = vmulq_f32(_v23, _v24);
                vst1q_f32(&term[_row][_col], _vr25);
                float32x4_t _v26 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr27 = vdivq_f32(_v26, _vs2);
                vst1q_f32(&term[_row][_col], _vr27);
                float32x4_t _v28 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v29 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr30 = vaddq_f32(_v28, _v29);
                vst1q_f32(&result[_row][_col], _vr30);
                float32x4_t _v31 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v32 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr33 = vmulq_f32(_v31, _v32);
                vst1q_f32(&term[_row][_col], _vr33);
                float32x4_t _v34 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr35 = vdivq_f32(_v34, _vs3);
                vst1q_f32(&term[_row][_col], _vr35);
                float32x4_t _v36 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v37 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr38 = vaddq_f32(_v36, _v37);
                vst1q_f32(&result[_row][_col], _vr38);
                float32x4_t _v39 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v40 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr41 = vmulq_f32(_v39, _v40);
                vst1q_f32(&term[_row][_col], _vr41);
                float32x4_t _v42 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr43 = vdivq_f32(_v42, _vs4);
                vst1q_f32(&term[_row][_col], _vr43);
                float32x4_t _v44 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v45 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr46 = vaddq_f32(_v44, _v45);
                vst1q_f32(&result[_row][_col], _vr46);
                float32x4_t _v47 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v48 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr49 = vmulq_f32(_v47, _v48);
                vst1q_f32(&term[_row][_col], _vr49);
                float32x4_t _v50 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr51 = vdivq_f32(_v50, _vs5);
                vst1q_f32(&term[_row][_col], _vr51);
                float32x4_t _v52 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v53 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr54 = vaddq_f32(_v52, _v53);
                vst1q_f32(&result[_row][_col], _vr54);
                float32x4_t _v55 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v56 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr57 = vmulq_f32(_v55, _v56);
                vst1q_f32(&term[_row][_col], _vr57);
                float32x4_t _v58 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr59 = vdivq_f32(_v58, _vs6);
                vst1q_f32(&term[_row][_col], _vr59);
                float32x4_t _v60 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v61 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr62 = vaddq_f32(_v60, _v61);
                vst1q_f32(&result[_row][_col], _vr62);
                float32x4_t _vs63 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 128 + _col], _vs63);
            }
            // Scalar cleanup
            for (; _col < 128; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 128 + _col];
                result[_row][_col] = x[_row][_col] * 1.0f;
                x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
                term[_row][_col] = x[_row][_col] * 1.0f;
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 6.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 20.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 42.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 72.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 110.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 156.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                output[(tile_idx) * 4096 + _row * 128 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (23 ops): x=TLOAD(input,num_full_tiles,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,110.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,156.0f); result=TADD(result,term); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs64 = vdupq_n_f32(1.0f);
        float32x4_t _vs65 = vdupq_n_f32(6.0f);
        float32x4_t _vs66 = vdupq_n_f32(20.0f);
        float32x4_t _vs67 = vdupq_n_f32(42.0f);
        float32x4_t _vs68 = vdupq_n_f32(72.0f);
        float32x4_t _vs69 = vdupq_n_f32(110.0f);
        float32x4_t _vs70 = vdupq_n_f32(156.0f);
        for (int _row = 0; _row < 32; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 128; _col += 4) {
                float32x4_t _vl71 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 128 + _col]);
                vst1q_f32(&x[_row][_col], _vl71);
                float32x4_t _v72 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr73 = vmulq_f32(_v72, _vs64);
                vst1q_f32(&result[_row][_col], _vr73);
                float32x4_t _v74 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v75 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr76 = vmulq_f32(_v74, _v75);
                vst1q_f32(&x_squared[_row][_col], _vr76);
                float32x4_t _v77 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr78 = vmulq_f32(_v77, _vs64);
                vst1q_f32(&term[_row][_col], _vr78);
                float32x4_t _v79 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v80 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr81 = vmulq_f32(_v79, _v80);
                vst1q_f32(&term[_row][_col], _vr81);
                float32x4_t _v82 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr83 = vdivq_f32(_v82, _vs65);
                vst1q_f32(&term[_row][_col], _vr83);
                float32x4_t _v84 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v85 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr86 = vaddq_f32(_v84, _v85);
                vst1q_f32(&result[_row][_col], _vr86);
                float32x4_t _v87 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v88 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr89 = vmulq_f32(_v87, _v88);
                vst1q_f32(&term[_row][_col], _vr89);
                float32x4_t _v90 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr91 = vdivq_f32(_v90, _vs66);
                vst1q_f32(&term[_row][_col], _vr91);
                float32x4_t _v92 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v93 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr94 = vaddq_f32(_v92, _v93);
                vst1q_f32(&result[_row][_col], _vr94);
                float32x4_t _v95 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v96 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr97 = vmulq_f32(_v95, _v96);
                vst1q_f32(&term[_row][_col], _vr97);
                float32x4_t _v98 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr99 = vdivq_f32(_v98, _vs67);
                vst1q_f32(&term[_row][_col], _vr99);
                float32x4_t _v100 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v101 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr102 = vaddq_f32(_v100, _v101);
                vst1q_f32(&result[_row][_col], _vr102);
                float32x4_t _v103 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v104 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr105 = vmulq_f32(_v103, _v104);
                vst1q_f32(&term[_row][_col], _vr105);
                float32x4_t _v106 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr107 = vdivq_f32(_v106, _vs68);
                vst1q_f32(&term[_row][_col], _vr107);
                float32x4_t _v108 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v109 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr110 = vaddq_f32(_v108, _v109);
                vst1q_f32(&result[_row][_col], _vr110);
                float32x4_t _v111 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v112 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr113 = vmulq_f32(_v111, _v112);
                vst1q_f32(&term[_row][_col], _vr113);
                float32x4_t _v114 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr115 = vdivq_f32(_v114, _vs69);
                vst1q_f32(&term[_row][_col], _vr115);
                float32x4_t _v116 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v117 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr118 = vaddq_f32(_v116, _v117);
                vst1q_f32(&result[_row][_col], _vr118);
                float32x4_t _v119 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v120 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr121 = vmulq_f32(_v119, _v120);
                vst1q_f32(&term[_row][_col], _vr121);
                float32x4_t _v122 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr123 = vdivq_f32(_v122, _vs70);
                vst1q_f32(&term[_row][_col], _vr123);
                float32x4_t _v124 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v125 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr126 = vaddq_f32(_v124, _v125);
                vst1q_f32(&result[_row][_col], _vr126);
                float32x4_t _vs127 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 128 + _col], _vs127);
            }
            // Scalar cleanup
            for (; _col < 128; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 128 + _col];
                result[_row][_col] = x[_row][_col] * 1.0f;
                x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
                term[_row][_col] = x[_row][_col] * 1.0f;
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 6.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 20.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 42.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 72.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 110.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
                term[_row][_col] = term[_row][_col] / 156.0f;
                result[_row][_col] = result[_row][_col] + term[_row][_col];
                output[(num_full_tiles) * 4096 + _row * 128 + _col] = result[_row][_col];
            }
        }

    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "sinh_taylor"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
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
    sinh_taylor((float*)args[0], (float*)args[1], (int32_t)0, (int32_t)0, (int32_t)0, (int32_t)0);
}
#endif  // PTO_CPU_SMOKE_RUNNER