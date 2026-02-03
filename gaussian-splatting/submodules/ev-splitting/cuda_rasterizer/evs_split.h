#pragma once
#include <cstdint>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
__global__ void EVS_split_kernel(
    int P,
    int num_splits,  // 总共要 split 的数量，用于边界检查
    const float* means3D,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const uint32_t* split_flag,  // const: benefit-cost check is done in preprocess
    const uint32_t* split_offsets,
    int n_clips,
    const float* clippers,
    // Note: benefit-cost parameters removed - check is now done in preprocess
    float* out_means3D,
    float* out_scales,
    float* out_rotations,
    float* out_opacities,
    uint32_t* out_src_indices
);
