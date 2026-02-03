/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool rr_clipping,
		int n_clips ,
		float* clippers,

			// add vars for cropper
		uint32_t* split_flag,  // size = P
		bool clip_model,
		// Benefit-cost split control parameters
		int evs_split_mode,      // 0=naive (split all), 1=proxy_control (benefit-cost)
		int evs_cost_mode,       // 0=1-min(Cl,Cr), 1=|Cl-Cr|
		float evs_lambda         // benefit-cost threshold
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* depths,
		const float4* conic_opacity,
		float* out_alpha,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_depth,
		/*
		add some data for ray tracing
		*/
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* cam_pos,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* viewmatrix,
		const float* projmatrix,
		float* debug_values,
		bool rr_clipping,
		bool rr_strategy,
		bool oenD_gs_strategy,
		int n_clips ,
		float* clippers,
		bool vizPlane,
		int n_inters ,
		float* intersections_tensor,

			// add vars for cropper
		uint32_t* split_flag  // size = P
	);
}


#endif