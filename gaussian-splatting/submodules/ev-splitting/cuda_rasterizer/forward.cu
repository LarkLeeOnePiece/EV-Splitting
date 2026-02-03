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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ bool inverse3x3_slow_dynamic(const float* A, float* Ainv, int N = 3) {
    // 增广矩阵，使用指针模拟动态行为
    float aug[3][6];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            aug[i][j + N] = (i == j) ? 1.f : 0.f;
        }
    }

    // 高斯消元部分（动态 N 控制）
    for (int i = 0; i < N; ++i) {
        float pivot = aug[i][i];
        if (fabsf(pivot) < 1e-6f) return false;

        for (int j = 0; j < 2 * N; ++j) {
            aug[i][j] /= pivot;
        }

        for (int k = 0; k < N; ++k) {
            if (k == i) continue;
            float factor = aug[k][i];
            for (int j = 0; j < 2 * N; ++j) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // 拷贝逆矩阵部分
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ainv[i * N + j] = aug[i][j + N];
        }
    }

    return true;
}


__device__ bool inverse3x3_slow(const float* A, float* Ainv) {
    // 创建增广矩阵 3x6: 左边是 A，右边是单位阵
    float aug[3][6];
    for (int i = 0; i < 3; ++i) {
        aug[i][0] = A[i * 3 + 0];
        aug[i][1] = A[i * 3 + 1];
        aug[i][2] = A[i * 3 + 2];
        aug[i][3] = (i == 0) ? 1.f : 0.f;
        aug[i][4] = (i == 1) ? 1.f : 0.f;
        aug[i][5] = (i == 2) ? 1.f : 0.f;
    }

    // 消元
    for (int i = 0; i < 3; ++i) {
        // 主元为 aug[i][i]，做非零检查
        float pivot = aug[i][i];
        if (fabsf(pivot) < 1e-8f) return false;

        // 归一化该行
        for (int j = 0; j < 6; ++j) aug[i][j] /= pivot;

        // 用该行消去其他行的第 i 列
        for (int k = 0; k < 3; ++k) {
            if (k == i) continue;
            float factor = aug[k][i];
            for (int j = 0; j < 6; ++j) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // 复制右边的 3x3 到 Ainv
    for (int i = 0; i < 3; ++i) {
        Ainv[i * 3 + 0] = aug[i][3];
        Ainv[i * 3 + 1] = aug[i][4];
        Ainv[i * 3 + 2] = aug[i][5];
    }

    return true;
}

__device__ bool inverse3x3(const float* A, float* Ainv) {
    // 输入:  A[9] 原始矩阵 (row-major)
    // 输出: Ainv[9] 逆矩阵 (row-major)
    
    float a = A[0], b = A[1], c = A[2];
    float d = A[3], e = A[4], f = A[5];
    float g = A[6], h = A[7], i = A[8];

    // 计算行列式
    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);

    if (fabsf(det) < 1e-8f) {
        // 不可逆，返回 false
        return false;
    }

    float invDet = 1.0f / det;

    // 伴随矩阵乘以 1/det 得到逆矩阵（cofactor 转置）
    Ainv[0] =  (e * i - f * h) * invDet;
    Ainv[1] = -(b * i - c * h) * invDet;
    Ainv[2] =  (b * f - c * e) * invDet;

    Ainv[3] = -(d * i - f * g) * invDet;
    Ainv[4] =  (a * i - c * g) * invDet;
    Ainv[5] = -(a * f - c * d) * invDet;

    Ainv[6] =  (d * h - e * g) * invDet;
    Ainv[7] = -(a * h - b * g) * invDet;
    Ainv[8] =  (a * e - b * d) * invDet;

    return true;
}
__device__ void compute1DGS(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& scale,
	float mod,
	const float4& rot,
    float& mu_t,
    float& sigma_t)
{
	float3 delta = make_float3(ray_origin.x - mu.x,
                                ray_origin.y - mu.y,
                                ray_origin.z - mu.z);
	// 1. 计算 Sigma
	float Sigma[9];
	computeSigma(Sigma, scale, mod, rot);  // 这是你的已有函数

	// 2. 使用通用矩阵求逆（慢）
	float invSigma[9];
	int dynamic_N = threadIdx.x % 5 + 1;  //
	dynamic_N = 3;  // 
	bool ok = inverse3x3_slow_dynamic(Sigma, invSigma,dynamic_N);
	if (!ok) {
		// 处理不可逆情况
		mu_t = 0.0;
    	sigma_t = 0.0;
	}

	// 3. 用 invSigma 计算 A 和 B
	float3 Su = mat3x3TimesVec3(ray_dir, invSigma);
	float3 Sd = mat3x3TimesVec3(delta, invSigma);

	float A = float3dot(Su, ray_dir);
	float B = float3dot(Sd, ray_dir);

    mu_t = -B / A;
    sigma_t = 1.f / sqrtf(A);
}

__device__ float interval_probability_erf(float mean, float sigma, float a, float b) {
    // in case of 0
    if (sigma <= 1e-6f) return (a <= mean && mean <= b) ? 1.0f : 0.0f;

    float sqrt2 = 1.4142135623730951f;  // CUDART_SQRT_TWO
    float z1 = (a - mean) / (sigma * sqrt2);
    float z2 = (b - mean) / (sigma * sqrt2);

    float result = 0.5 * (erf(z2) - erf(z1));
    return result;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}





// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool rr_clipping,
	int n_clips ,
	float* clippers,

	//cropper vars
	uint32_t* split_flag,  // size = P
	bool clip_model,
	// Benefit-cost split control parameters
	int evs_split_mode,      // 0=naive (split all), 1=proxy_control (benefit-cost)
	int evs_cost_mode,       // 0=1-min(Cl,Cr), 1=|Cl-Cr|
	float evs_lambda         // benefit-cost threshold

)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	split_flag[idx] = 0;

	// Transform point - needed for both clipping and frustum culling
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// ============================================================
	// FIRST: Calculate split_flag based on clipping planes
	// This is independent of camera view - same Gaussian should
	// always have the same split behavior regardless of viewpoint
	// ============================================================
	if (n_clips > 0) {
		bool clipped_out = false;  // Track if Gaussian should be culled (for clip_model mode)
		bool need_split = false;
		for (int clps = 0; clps < n_clips; clps++) {
			float3 p_normal = { clippers[4 * clps], clippers[4 * clps + 1], clippers[4 * clps + 2] };
			float p_dis = clippers[4 * clps + 3];

			// Check if Gaussian is on the wrong side of the clipping plane
			bool vis_flag = is_visible(p_orig, p_normal, p_dis);

			if (rr_clipping) {
				float d = point_to_plane_distance(p_orig, p_normal, p_dis);
				float scale_x = scales[idx].x * scale_modifier;
				float scale_y = scales[idx].y * scale_modifier;
				float scale_z = scales[idx].z * scale_modifier;
				float sigma_max = fmaxf(fmaxf(scale_x, scale_y), scale_z);

				bool cross_plane = fabs(d) <= 3.0f * sigma_max;

				if (cross_plane) {
					bool should_split = true;  // Default: naive mode splits all

					// Benefit-cost check for proxy control mode
					if (evs_split_mode == 1) {
						// Compute 3D covariance matrix
						float cov3D_local[6];
						computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D_local);

						// Unpack to 3x3 symmetric matrix (row-major)
						float Sigma0[9] = {
							cov3D_local[0], cov3D_local[1], cov3D_local[2],
							cov3D_local[1], cov3D_local[3], cov3D_local[4],
							cov3D_local[2], cov3D_local[4], cov3D_local[5]
						};

						// L0 = Sigma0 * n (plane normal)
						float n[3] = {p_normal.x, p_normal.y, p_normal.z};
						float L0[3] = {
							Sigma0[0]*n[0] + Sigma0[1]*n[1] + Sigma0[2]*n[2],
							Sigma0[3]*n[0] + Sigma0[4]*n[1] + Sigma0[5]*n[2],
							Sigma0[6]*n[0] + Sigma0[7]*n[1] + Sigma0[8]*n[2]
						};

						// tau^2 = n^T * Sigma0 * n (variance along plane normal)
						float tau2 = n[0]*L0[0] + n[1]*L0[1] + n[2]*L0[2];
						float tau = sqrtf(fmaxf(tau2, 1e-8f));

						// d0 = signed distance to plane (already computed as 'd')
						float d0 = d;

						// Cl, Cr using error function
						float t = d0 / (1.41421356f * tau);
						float Cl = 0.5f * (1.0f - erff(t));
						float Cr = 1.0f - Cl;
						Cl = fmaxf(Cl, 1e-6f);
						Cr = fmaxf(Cr, 1e-6f);

						// Benefit: higher when closer to plane
						float benefit = expf(-0.5f * d0 * d0 / (tau * tau));

						// Cost: based on split asymmetry
						float cost;
						if (evs_cost_mode == 0) {
							cost = 1.0f - fminf(Cl, Cr);  // Mode 0: Asymmetry penalty
						} else {
							cost = fabsf(Cl - Cr);         // Mode 1: Conservative
						}

						// Decision: only split if benefit > lambda * cost
						should_split = (benefit > evs_lambda * cost);
					}

					if (should_split) {
						need_split = true;
					}
				}

				// In clip_model mode, cull Gaussians that don't cross the plane
				if (clip_model && !vis_flag) {
					clipped_out = true;
				}
			} else {
				// use the mean to judge the visible
				float d = point_to_plane_distance(p_orig, p_normal, p_dis);
				float scale_x = scales[idx].x * scale_modifier;
				float scale_y = scales[idx].y * scale_modifier;
				float scale_z = scales[idx].z * scale_modifier;
				float sigma_max = fmaxf(fmaxf(scale_x, scale_y), scale_z);

				bool cross_plane = fabs(d) <= 3.0f * sigma_max;

				if (cross_plane) {
					bool should_split = true;  // Default: naive mode splits all

					// Benefit-cost check for proxy control mode
					if (evs_split_mode == 1) {
						// Compute 3D covariance matrix
						float cov3D_local[6];
						computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D_local);

						// Unpack to 3x3 symmetric matrix (row-major)
						float Sigma0[9] = {
							cov3D_local[0], cov3D_local[1], cov3D_local[2],
							cov3D_local[1], cov3D_local[3], cov3D_local[4],
							cov3D_local[2], cov3D_local[4], cov3D_local[5]
						};

						// L0 = Sigma0 * n (plane normal)
						float n[3] = {p_normal.x, p_normal.y, p_normal.z};
						float L0[3] = {
							Sigma0[0]*n[0] + Sigma0[1]*n[1] + Sigma0[2]*n[2],
							Sigma0[3]*n[0] + Sigma0[4]*n[1] + Sigma0[5]*n[2],
							Sigma0[6]*n[0] + Sigma0[7]*n[1] + Sigma0[8]*n[2]
						};

						// tau^2 = n^T * Sigma0 * n (variance along plane normal)
						float tau2 = n[0]*L0[0] + n[1]*L0[1] + n[2]*L0[2];
						float tau = sqrtf(fmaxf(tau2, 1e-8f));

						// d0 = signed distance to plane (already computed as 'd')
						float d0 = d;

						// Cl, Cr using error function
						float t = d0 / (1.41421356f * tau);
						float Cl = 0.5f * (1.0f - erff(t));
						float Cr = 1.0f - Cl;
						Cl = fmaxf(Cl, 1e-6f);
						Cr = fmaxf(Cr, 1e-6f);

						// Benefit: higher when closer to plane
						float benefit = expf(-0.5f * d0 * d0 / (tau * tau));

						// Cost: based on split asymmetry
						float cost;
						if (evs_cost_mode == 0) {
							cost = 1.0f - fminf(Cl, Cr);  // Mode 0: Asymmetry penalty
						} else {
							cost = fabsf(Cl - Cr);         // Mode 1: Conservative
						}

						// Decision: only split if benefit > lambda * cost
						should_split = (benefit > evs_lambda * cost);
					}

					if (should_split) {
						need_split = true;
					}
				}

				// In clip_model mode, cull Gaussians that are not visible
				if (clip_model && !vis_flag) {
					clipped_out = true;
				}

				// 最后统一写 split_flag
			}

			if (need_split) {
					split_flag[idx] = 1;
				}
		}

		// If clip_model is enabled and Gaussian is culled, set radius to 0 and return
		if (clip_model && clipped_out) {
			return;
		}
	}

	// ============================================================
	// SECOND: Frustum culling (depends on camera view)
	// ============================================================
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// ============================================================
	// THIRD: Continue with rendering pipeline
	// ============================================================




	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* __restrict__ cam_pos,
	const float* __restrict__ orig_points,
	const glm::vec3* __restrict__ scales,
	const float scale_modifier,
	const glm::vec4* __restrict__ rotations,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float* __restrict__ debug_values,
	bool rr_clipping,
	bool rr_strategy,
	bool oenD_gs_strategy,
	int n_clips ,
	const float* clippers,
	bool vizPlane,
	int n_inters ,
	const float* intersections_tensor,

		// add vars for cropper
	uint32_t* split_flag  // size = P
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;


	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	__shared__ float3 collected_means[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];
	__shared__ float4 collected_rots[BLOCK_SIZE];


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];


			collected_means[block.thread_rank()]={orig_points[3 * coll_id], orig_points[3 * coll_id + 1], orig_points[3 * coll_id + 2]};
			collected_scales[block.thread_rank()]={scales[coll_id].x,scales[coll_id].y,scales[coll_id].z};
			collected_rots[block.thread_rank()]={rotations[coll_id].x,rotations[coll_id].y,rotations[coll_id].z,rotations[coll_id].w};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			float decay_weight=0.0f;// decay the power, I need  to set it as 0 for initilization, otherwise have some werid case for the intersection! need to fix
			if (power > 0.0f) continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			float random=(float(collected_id[j])+100.0)/(float(collected_id[j])*2.0);
			// float3 fakeRGB={}

			for (int ch = 0; ch < CHANNELS; ch++){
					// add some debug code for split
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}
				
			D += depths[collected_id[j]] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		out_alpha[pix_id] = 1 - T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;



		// render the line in the end, in case of the overwrite
		if (vizPlane){
			for(int i=0;i<n_inters-1;i++){
				float3 p_orig = { intersections_tensor[3 * i], intersections_tensor[3 * i + 1], intersections_tensor[3 * i + 2] };
				float4 p_hom = transformPoint4x4(p_orig, projmatrix);
				float p_w = 1.0f / (p_hom.w + 0.0000001f);
				float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_start = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				int end=i+1;
				p_orig = { intersections_tensor[3 * end], intersections_tensor[3 * end + 1], intersections_tensor[3 * end + 2] };
				p_hom = transformPoint4x4(p_orig, projmatrix);
				p_w = 1.0f / (p_hom.w + 0.0000001f);
				p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_end = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				if (pixel_on_line(pixf, point_image_start, point_image_end,0.5f)) {
					// 写入像素值，比如 framebuffer[pix_id] = 1;
					out_color[0 * H * W + pix_id]=1.0f;
					out_color[1 * H * W + pix_id]=0.0f;
					out_color[2 * H * W + pix_id]=0.0f;// fill red 
				}
			}
				// for the last segment to have the close rectangle
				int i=n_inters-1;
				float3 p_orig = { intersections_tensor[3 * i], intersections_tensor[3 * i + 1], intersections_tensor[3 * i + 2] };
				float4 p_hom = transformPoint4x4(p_orig, projmatrix);
				float p_w = 1.0f / (p_hom.w + 0.0000001f);
				float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_start = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				int end=0;
				p_orig = { intersections_tensor[3 * end], intersections_tensor[3 * end + 1], intersections_tensor[3 * end + 2] };
				p_hom = transformPoint4x4(p_orig, projmatrix);
				p_w = 1.0f / (p_hom.w + 0.0000001f);
				p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
				float2 point_image_end = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

				if (pixel_on_line(pixf, point_image_start, point_image_end,0.5f)) {
					// 写入像素值，比如 framebuffer[pix_id] = 1;
					out_color[0 * H * W + pix_id]=1.0f;
					out_color[1 * H * W + pix_id]=0.0f;
					out_color[2 * H * W + pix_id]=0.0f;// fill red 
				}
		}

	}


}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	float* out_alpha,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
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
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		focal_x,focal_y,
		tan_fovx, tan_fovy,
		cam_pos,
		orig_points,
		scales,
		scale_modifier,
		rotations,
		viewmatrix,
		projmatrix,
		debug_values,
		rr_clipping,
		rr_strategy,
		oenD_gs_strategy,
		n_clips,
		clippers,
		vizPlane,
		n_inters ,
		intersections_tensor,

		//cropper vars
		split_flag
	);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
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
	int evs_split_mode,
	int evs_cost_mode,
	float evs_lambda

)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rr_clipping,
		n_clips ,
		clippers,

		//crop vars
		split_flag,
		clip_model,
		// Benefit-cost split control parameters
		evs_split_mode,
		evs_cost_mode,
		evs_lambda
		);
}




