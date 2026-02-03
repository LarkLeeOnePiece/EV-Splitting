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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


#include "evs_data.h"
#include "evs_split.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	// DL: We need to add more variables here to store the raotation and scale for ray tracing render

	obtain(chunk, geom.split_flag,   P, 128); 

	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}


EVSSplitOutput CudaRasterizer::Rasterizer::forward_evs_split(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_alpha,
			int* radii ,
			bool debug ,
			//DL ： add some parameters for clipping
			bool rr_clipping ,
			bool rr_strategy ,
			bool oenD_gs_strategy ,
			int n_clips ,
			float* clippers ,
			bool vizPlane,
			int n_inters,
			float* intersections_tensor,
			bool clip_model,
			// Benefit-cost split control parameters
			int evs_split_mode,
			int evs_cost_mode,
			float evs_lambda
)
{
    // ------------------------------------------------------------
    // 1. Geometry state
    // ------------------------------------------------------------
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid(
        (width  + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y,
        1
    );

    // image buffer（虽然 split 不用，但接口保持一致）
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);


	cudaMemset(geomState.split_flag, 0, P * sizeof(uint32_t));

    // ------------------------------------------------------------
    // 2. Preprocess（唯一职责：正确写 split_flag）
    // ------------------------------------------------------------
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        width  / (2.0f * tan_fovx),
        height / (2.0f * tan_fovy),
        tan_fovx, tan_fovy,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered,
        rr_clipping,
        n_clips,
        clippers,
        geomState.split_flag,
        clip_model,
        // Benefit-cost split control parameters
        evs_split_mode,
        evs_cost_mode,
        evs_lambda
    ), debug);

    // ------------------------------------------------------------
    // 3. 统计 num_splits（✔ 正确方式：DeviceReduce）
    // ------------------------------------------------------------
    int* d_num_splits = nullptr;
    cudaMalloc(&d_num_splits, sizeof(int));
    cudaMemset(d_num_splits, 0, sizeof(int));

    void* reduce_temp = nullptr;
    size_t reduce_bytes = 0;

    cub::DeviceReduce::Sum(
        nullptr, reduce_bytes,
        geomState.split_flag,
        d_num_splits,
        P
    );

    cudaMalloc(&reduce_temp, reduce_bytes);

    cub::DeviceReduce::Sum(
        reduce_temp, reduce_bytes,
        geomState.split_flag,
        d_num_splits,
        P
    );

    int num_splits = 0;
    cudaMemcpy(&num_splits, d_num_splits, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_num_splits);
    cudaFree(reduce_temp);

    // ------------------------------------------------------------
    // 4. Prefix sum（只用于定位）
    // ------------------------------------------------------------
    uint32_t* split_offsets = nullptr;
    cudaMalloc(&split_offsets, P * sizeof(uint32_t));
    cudaMemset(split_offsets, 0, P * sizeof(uint32_t));

    void* scan_temp = nullptr;
    size_t scan_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes,
        geomState.split_flag,
        split_offsets,
        P
    );

    cudaMalloc(&scan_temp, scan_bytes);

    cub::DeviceScan::ExclusiveSum(
        scan_temp, scan_bytes,
        geomState.split_flag,
        split_offsets,
        P
    );

    cudaFree(scan_temp);

    // ------------------------------------------------------------
    // 5. Allocate outputs
    // ------------------------------------------------------------
    EVSSplitOutput out;
    out.P            = P;
    out.num_splits   = num_splits;
    out.split_flag   = geomState.split_flag;
    out.split_offsets = split_offsets;

    cudaMalloc(&out.means3D,    2 * num_splits * 3 * sizeof(float));
    cudaMalloc(&out.scales,     2 * num_splits * 3 * sizeof(float));
    cudaMalloc(&out.rotations,  2 * num_splits * 4 * sizeof(float));
    cudaMalloc(&out.opacities,  2 * num_splits * sizeof(float));
    cudaMalloc(&out.src_indices, num_splits * sizeof(uint32_t));

    // ------------------------------------------------------------
    // 6. EVS split kernel（安全）
    // Note: num_splits is now accurate because benefit-cost check is in preprocess
    // ------------------------------------------------------------
    int threads = 256;
    int blocks  = (P + threads - 1) / threads;

    EVS_split_kernel<<<blocks, threads>>>(
        P,
        num_splits,
        means3D,
        scales,
        rotations,
        opacities,
        geomState.split_flag,
        split_offsets,
        n_clips,
        clippers,
        // Note: benefit-cost parameters removed - check is now done in preprocess
        out.means3D,
        out.scales,
        out.rotations,
        out.opacities,
        out.src_indices
    );

    return out;
}

// {
// 	const float focal_y = height / (2.0f * tan_fovy);
// 	const float focal_x = width / (2.0f * tan_fovx);

// 	size_t chunk_size = required<GeometryState>(P);
// 	char* chunkptr = geometryBuffer(chunk_size);
// 	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

// 	if (radii == nullptr)
// 	{
// 		radii = geomState.internal_radii;
// 	}

// 	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
// 	dim3 block(BLOCK_X, BLOCK_Y, 1);

// 	// Dynamically resize image-based auxiliary buffers during training
// 	size_t img_chunk_size = required<ImageState>(width * height);
// 	char* img_chunkptr = imageBuffer(img_chunk_size);
// 	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

// 	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
// 	{
// 		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
// 	}


// 	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
// 	CHECK_CUDA(FORWARD::preprocess(
// 		P, D, M,
// 		means3D,
// 		(glm::vec3*)scales,
// 		scale_modifier,
// 		(glm::vec4*)rotations,
// 		opacities,
// 		shs,
// 		geomState.clamped,
// 		cov3D_precomp,
// 		colors_precomp,
// 		viewmatrix, projmatrix,
// 		(glm::vec3*)cam_pos,
// 		width, height,
// 		focal_x, focal_y,
// 		tan_fovx, tan_fovy,
// 		radii,
// 		geomState.means2D,
// 		geomState.depths,
// 		geomState.cov3D,
// 		geomState.rgb,
// 		geomState.conic_opacity,
// 		tile_grid,
// 		geomState.tiles_touched,
// 		prefiltered,
// 		rr_clipping,
// 		n_clips ,
// 		clippers,

// 		geomState.split_flag,
// 		clip_model
// 	), debug)

//     // ------------------------------
//     // 3. Prefix sum on split_flag
//     // ------------------------------
//     uint32_t* split_offsets = nullptr;
//     cudaMalloc(&split_offsets, P * sizeof(uint32_t));

//     // 清零 split_offsets，防止旧数据影响 prefix sum
//     cudaMemset(split_offsets, 0, P * sizeof(uint32_t));

//     void* scan_temp = nullptr;
//     size_t scan_bytes = 0;

//     cub::DeviceScan::ExclusiveSum(
//         nullptr, scan_bytes,
//         geomState.split_flag,
//         split_offsets,
//         P
//     );

//     cudaMalloc(&scan_temp, scan_bytes);

//     cub::DeviceScan::ExclusiveSum(
//         scan_temp, scan_bytes,
//         geomState.split_flag,
//         split_offsets,
//         P
//     );

//     uint8_t  last_flag;
//     uint32_t last_offset;

//     cudaMemcpy(&last_flag,
//         geomState.split_flag + P - 1,
//         sizeof(uint8_t),
//         cudaMemcpyDeviceToHost);

//     cudaMemcpy(&last_offset,
//         split_offsets + P - 1,
//         sizeof(uint32_t),
//         cudaMemcpyDeviceToHost);

//     int num_splits = last_offset + last_flag;

//     // ------------------------------
//     // 4. Allocate output buffers
//     // ------------------------------
//     EVSSplitOutput out;
//     out.P          = P;
//     out.num_splits = num_splits;
//     out.split_flag    = geomState.split_flag;
//     out.split_offsets = split_offsets;
	

// 	// std::cout << "[CUDA CudaRasterizer::Rasterizer::forward_evs_split Debug] num_splits = " 
// 	// 		<< out.num_splits << std::endl;

//     cudaMalloc(&out.means3D,   2 * num_splits * 3 * sizeof(float));
//     cudaMalloc(&out.scales,    2 * num_splits * 3 * sizeof(float));
//     cudaMalloc(&out.rotations, 2 * num_splits * 4 * sizeof(float));
//     cudaMalloc(&out.opacities, 2 * num_splits * sizeof(float));
//     cudaMalloc(&out.src_indices, num_splits * sizeof(uint32_t));

//     // ------------------------------
//     // 5. EVS split kernel
//     // ------------------------------
//     int threads = 256;
//     int blocks  = (P + threads - 1) / threads;

//     EVS_split_kernel<<<blocks, threads>>>(
//         P,
//         num_splits,  // 传入 num_splits 用于边界检查
//         means3D,
//         scales,
//         rotations,
//         opacities,
//         geomState.split_flag,
//         split_offsets,
//         n_clips,
//         clippers,
//         out.means3D,
//         out.scales,
//         out.rotations,
//         out.opacities,
//         out.src_indices
//     );

//     cudaFree(scan_temp);

//     return out;
// }


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	float* out_alpha,
	int* radii,
	bool debug,
	bool rr_clipping,
	bool rr_strategy,
	bool oenD_gs_strategy,
	int n_clips ,
	float* clippers,

	bool vizPlane,
	int n_inters,
	float* intersections_tensor,
	bool clip_model
)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}


	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		rr_clipping,
		n_clips ,
		clippers,

		geomState.split_flag,
		clip_model,
		// Benefit-cost split control parameters (defaults: naive mode)
		0,    // evs_split_mode = 0 (naive)
		0,    // evs_cost_mode = 0
		0.0f  // evs_lambda = 0.0
	), debug)


	// //--------------------------------------------------
	// // from now on, I want to setup the code only for evs split

	// uint32_t* split_offsets;  // [P]
	// cudaMalloc(&split_offsets, P * sizeof(uint32_t));

	// // 清零 split_offsets，防止旧数据影响 prefix sum
	// cudaMemset(split_offsets, 0, P * sizeof(uint32_t));

	// void* scan_temp = nullptr;
	// size_t scan_bytes = 0;

	// // dry run
	// cub::DeviceScan::ExclusiveSum(
	// 	scan_temp, scan_bytes,
	// 	geomState.split_flag,
	// 	split_offsets,
	// 	P
	// );

	// // allocate
	// cudaMalloc(&scan_temp, scan_bytes);

	// // real scan
	// cub::DeviceScan::ExclusiveSum(
	// 	scan_temp, scan_bytes,
	// 	geomState.split_flag,
	// 	split_offsets,
	// 	P
	// );

	// uint8_t  last_flag;
	// uint32_t last_offset;

	// cudaMemcpy(&last_flag,
	// 		geomState.split_flag + P - 1,
	// 		sizeof(uint8_t),
	// 		cudaMemcpyDeviceToHost);

	// cudaMemcpy(&last_offset,
	// 		split_offsets + P - 1,
	// 		sizeof(uint32_t),
	// 		cudaMemcpyDeviceToHost);

	// int num_splits = last_offset + last_flag;


	// SplitResult result;
	// result.num_splits = num_splits;

	// cudaMalloc(&result.means3D,   2 * num_splits * 3 * sizeof(float));
	// cudaMalloc(&result.scales,    2 * num_splits * 3 * sizeof(float));
	// cudaMalloc(&result.rotations, 2 * num_splits * 4 * sizeof(float));
	// cudaMalloc(&result.opacities, 2 * num_splits * sizeof(float));

	// cudaMalloc(&result.src_indices, num_splits * sizeof(uint32_t));


	// int threads = 256;
	// int blocks = (P + threads - 1) / threads;

	// EVS_split_kernel<<<blocks, threads>>>(
	// 	P,
	// 	num_splits,  // 传入 num_splits 用于边界检查
	// 	means3D,
	// 	scales,
	// 	rotations,
	// 	opacities,
	// 	geomState.split_flag,
	// 	split_offsets,
	// 	n_clips,
	// 	clippers,
	// 	result.means3D,
	// 	result.scales,
	// 	result.rotations,
	// 	result.opacities,
	// 	result.src_indices
	// );

	// //--------------------------------------------------

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8] # DL: [2, 5, 5, 7, 8] in geomState.point_offsets indicating the tiles for each gaussian
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	
	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// std::cout<<" clipping in world space num_rendered:"<<num_rendered<<std::endl;

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	
	// save the media data
	int N=50;
	float* h_data = new float[N];  // N 是你需要的元素个数
	for (int i = 0; i < N; ++i) h_data[i] =0.0f;  // 初始化数据
	float* debug_values;
	cudaMalloc(&debug_values, N * sizeof(float));
	cudaMemcpy(debug_values, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

	
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		out_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		// foloowing are for ray tracing
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)cam_pos,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		viewmatrix, 
		projmatrix,
		debug_values,
		rr_clipping,
		rr_strategy,
		oenD_gs_strategy,
		n_clips ,
		clippers,
		vizPlane,
		n_inters,
		intersections_tensor,

		geomState.split_flag
	), debug)

	// cudaMemcpy(h_data, debug_values, N * sizeof(float), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < N; ++i) {
	// 	std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
	// }

	// delete[] h_data;
	// cudaFree(debug_values);


	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* accum_alphas,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	const float* dL_dpix_alpha,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		accum_alphas,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		dL_dpix_alpha,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}