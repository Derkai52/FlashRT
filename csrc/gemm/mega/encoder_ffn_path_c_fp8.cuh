#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Path C FP8 encoder FFN (GateUp → GeGLU+quant → Down → res+next-rms).
// mode=0 ref:   identical 4-step production sequence (full Se×H hid).
// mode=1 fused: same math; GeGLU+Down in M-strips so hid working set is
//               STRIP×H (default 256) instead of Se×H.

constexpr int kEncoderFfnPathCFp8StripM = 256;

extern "C" int encoder_ffn_path_c_fp8(
    void* x_fp8,
    void* gate_w,
    void* down_w,
    void* gate_scratch,
    void* hid_fp8,
    void* fg,
    void* x_resid,
    void* x_fp8_next,
    int Se,
    int H,
    int D,
    float alpha_gu,
    float alpha_down,
    const float* as_down,
    const float* as_next,
    int mode,
    cudaStream_t stream);
