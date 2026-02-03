#ifndef CUDA_RASTERIZER_EVS_DATA_H_INCLUDED
#define CUDA_RASTERIZER_EVS_DATA_H_INCLUDED

#pragma once
#include <cstdint>


struct SplitResult {
    int num_splits;            // S
    uint32_t* src_indices;     // [S]，原 Gaussian 的 idx
    float* means3D;            // [2*S, 3]
    float* scales;             // [2*S, 3]
    float* rotations;          // [2*S, 4]
    float* opacities;          // [2*S]
};


SplitResult split_gaussians_evs(
    int P,
    const float* means3D,
    const float* scales,
    const float* rotations,
    const float* opacities,

    const uint32_t* split_flag,   // [P]
    int n_clips,
    const float* clippers        // [n_clips,4]

    // stream / debug 可自行加
);

struct EVSSplitOutput
{
    // 原始 Gaussian 维度
    int P;

    // split 统计
    int num_splits;

    // --- device buffers ---
    uint32_t*  split_flag;     // [P]
    uint32_t* split_offsets;  // [P]

    // --- split results ---
    float* means3D;           // [2*num_splits, 3]
    float* scales;            // [2*num_splits, 3]
    float* rotations;         // [2*num_splits, 4]
    float* opacities;         // [2*num_splits]
    uint32_t* src_indices;    // [num_splits]
};

#endif