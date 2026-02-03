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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include <math_constants.h>  // for CUDART_SQRT_TWO
#include <math_functions.h>  // for erf()

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

// DL: DL function
__forceinline__ __device__  float3 normalize_float3(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-6f) {  
        return make_float3(v.x / len, v.y / len, v.z / len);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);  // 或者保留原值，按需要处理
    }
}

// DL: 
__forceinline__ __device__  float3 mat3x3TimesVec3(const float3& p, const float* matrix) {
	// input : point or vector, mat3x3
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[3] * p.x + matrix[4] * p.y + matrix[5] * p.z,
		matrix[6] * p.x + matrix[7] * p.y + matrix[8] * p.z,
	};
	return transformed;
}
__forceinline__ __device__  void computeT_mat(float* T,const float3& scale, float mod, const float4& rot) {
	// input : scale mod rot

	float4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	float R_ellipsoid_w[9]={
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	};

	float inv_sx = 1.0/(3.0*mod * scale.x);
	float inv_sy = 1.0/(3.0*mod * scale.y);
	float inv_sz = 1.0/(3.0*mod * scale.z);

	T[0]=R_ellipsoid_w[0]*inv_sx; T[1]=R_ellipsoid_w[3]*inv_sx; T[2]=R_ellipsoid_w[6]*inv_sx;
	T[3]=R_ellipsoid_w[1]*inv_sy; T[4]=R_ellipsoid_w[4]*inv_sy; T[5]=R_ellipsoid_w[7]*inv_sy;
	T[6]=R_ellipsoid_w[2]*inv_sz; T[7]=R_ellipsoid_w[5]*inv_sz; T[8]=R_ellipsoid_w[8]*inv_sz;
}


__forceinline__ __device__  void computeSigma(float* Sigma,const float3& scale, float mod, const float4& rot) {

	float4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	float R[9]={
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	};

	    // 2. compute scaled diagonal
    float sx = scale.x * mod;
    float sy = scale.y * mod;
    float sz = scale.z * mod;

    //float D[3] = { sx * sx, sy * sy, sz * sz };  // diag((s * m)^2)

	float dx2=sx * sx;
	float dy2=sy * sy;
	float dz2=sz * sz;

	Sigma[0] = dx2 * R[0] * R[0] + dy2 * R[1] * R[1] + dz2 * R[2] * R[2];
	Sigma[1] = dx2 * R[0] * R[3] + dy2 * R[1] * R[4] + dz2 * R[2] * R[5];
	Sigma[2] = dx2 * R[0] * R[6] + dy2 * R[1] * R[7] + dz2 * R[2] * R[8];

	Sigma[3] = dx2 * R[3] * R[0] + dy2 * R[4] * R[1] + dz2 * R[5] * R[2];
	Sigma[4] = dx2 * R[3] * R[3] + dy2 * R[4] * R[4] + dz2 * R[5] * R[5];
	Sigma[5] = dx2 * R[3] * R[6] + dy2 * R[4] * R[7] + dz2 * R[5] * R[8];

	Sigma[6] = dx2 * R[6] * R[0] + dy2 * R[7] * R[1] + dz2 * R[8] * R[2];
	Sigma[7] = dx2 * R[6] * R[3] + dy2 * R[7] * R[4] + dz2 * R[8] * R[5];
	Sigma[8] = dx2 * R[6] * R[6] + dy2 * R[7] * R[7] + dz2 * R[8] * R[8];

}



__device__ inline float dot3(const float* a, const float* b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__forceinline__ __device__ float float3dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
//
__forceinline__ __device__ float intersect_ray_plane(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& plane_normal,
    float d
) {
    float denom = float3dot(plane_normal, ray_dir);
    if (fabsf(denom) < 1e-6f) {
        return 0.0f;  // parallel
    }

    float t = -(float3dot(plane_normal, ray_origin) + d) / denom;
    return t > 0.0f ? t : 0.0f;
}

__forceinline__ __device__ float point_to_plane_distance(const float3& p, const float3& normal, float d) {
    // put p in n·p + d，abs is the distance
    return fabsf(p.x * normal.x + p.y * normal.y + p.z * normal.z + d);
}


__forceinline__ __device__ bool is_visible(const float3& p_cam, const float3& plane_normal, float d) {
    float val = p_cam.x * plane_normal.x + p_cam.y * plane_normal.y + p_cam.z * plane_normal.z;
    return (val + d) > 0.0f;
}

__host__ __device__ inline float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__forceinline__  __device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__forceinline__ __device__ float fast_inv_length(float3 v) {
    return rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline bool pixel_on_line(float2 pixf, float2 p0, float2 p1, float thickness) {
    float2 ab = make_float2(p1.x - p0.x, p1.y - p0.y);
    float2 ap = make_float2(pixf.x + 0.5f - p0.x, pixf.y + 0.5f - p0.y); // pixel center
    float ab_len = sqrtf(ab.x * ab.x + ab.y * ab.y);
    if (ab_len < 1e-5f) return false;

    float t = (ap.x * ab.x + ap.y * ab.y) / (ab_len * ab_len);
    t = fminf(fmaxf(t, 0.0f), 1.0f); // clamp to [0,1]
    float2 closest = make_float2(p0.x + ab.x * t, p0.y + ab.y * t);
    float dx = closest.x - (pixf.x + 0.5f);
    float dy = closest.y - (pixf.y + 0.5f);
    float dist2 = dx * dx + dy * dy;
    return dist2 <= thickness * thickness;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// if (p_orig.x < -0.0f){
	// 	return false;// DL: we se clipping plane here
	// }
	
	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false; 
	}

	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif