#include "evs_split.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// from here, we add some functions for evs splitting

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.

//DL: copy this from forward.cu
__device__ void evs_computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
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

__device__ inline void mat3Identity(float M[9]) {
    M[0]=1; M[1]=0; M[2]=0;
    M[3]=0; M[4]=1; M[5]=0;
    M[6]=0; M[7]=0; M[8]=1;
}

__device__ inline void jacobiRotate(
    float A[9], float V[9], int p, int q
) {
    if (fabsf(A[3*p+q]) < 1e-8f) return;

    float app = A[3*p+p];
    float aqq = A[3*q+q];
    float apq = A[3*p+q];

    float phi = 0.5f * atanf(2.0f * apq / (aqq - app));
    float c = cosf(phi);
    float s = sinf(phi);

    // rotate A
    for (int k = 0; k < 3; ++k) {
        float aik = A[3*p+k];
        float aqk = A[3*q+k];
        A[3*p+k] = c*aik - s*aqk;
        A[3*q+k] = s*aik + c*aqk;
    }
    for (int k = 0; k < 3; ++k) {
        float akp = A[3*k+p];
        float akq = A[3*k+q];
        A[3*k+p] = c*akp - s*akq;
        A[3*k+q] = s*akp + c*akq;
    }

    // rotate eigenvectors
    for (int k = 0; k < 3; ++k) {
        float vip = V[3*k+p];
        float viq = V[3*k+q];
        V[3*k+p] = c*vip - s*viq;
        V[3*k+q] = s*vip + c*viq;
    }
}

__device__ void eigenDecompositionSym3(
    const float A_in[9],
    float eigval[3],
    float eigvec[9]
) {
    // copy matrix
    float A[9];
    for (int i = 0; i < 9; ++i) A[i] = A_in[i];

    mat3Identity(eigvec);

    // fixed number of Jacobi sweeps
    #pragma unroll
    for (int iter = 0; iter < 5; ++iter) {
        jacobiRotate(A, eigvec, 0, 1);
        jacobiRotate(A, eigvec, 0, 2);
        jacobiRotate(A, eigvec, 1, 2);
    }

    eigval[0] = A[0];
    eigval[1] = A[4];
    eigval[2] = A[8];
}

__device__ void mat3ToQuat(
    const float R[9],
    float q[4]
) {
    float trace = R[0] + R[4] + R[8];

    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        float s = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        float s = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25f * s;
    }

    // normalize for safety
    float norm = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (norm > 1e-8f) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

__device__ void covToScaleRot(
	const float Sigma[9],
	float scale[3],
	float quat[4]
)
{
	float eigval[3];
	float eigvec[9];
	eigenDecompositionSym3(Sigma, eigval, eigvec);

	// scale = sqrt(eigenvalues)
	scale[0] = sqrtf(fmaxf(eigval[0], 1e-8f));
	scale[1] = sqrtf(fmaxf(eigval[1], 1e-8f));
	scale[2] = sqrtf(fmaxf(eigval[2], 1e-8f));

	// eigvec → quaternion（标准 rot-matrix → quat）
	mat3ToQuat(eigvec, quat);
}
__device__ inline void unpackCov3D(const float* c, float S[9])
{
	S[0]=c[0]; S[1]=c[1]; S[2]=c[2];
	S[3]=c[1]; S[4]=c[3]; S[5]=c[4];
	S[6]=c[2]; S[7]=c[4]; S[8]=c[5];
}

__global__ void EVS_split_kernel(
	int P,
	int num_splits,
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
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P || !split_flag[idx]) return;

	uint32_t k = split_offsets[idx];
	if (k >= num_splits) return;

	// === load Gaussian ===
	float mu[3] = {
		means3D[3*idx+0],
		means3D[3*idx+1],
		means3D[3*idx+2]
	};
	float alpha0 = opacities[idx];

	glm::vec3 scale0(
		scales[3*idx+0],
		scales[3*idx+1],
		scales[3*idx+2]
	);
	glm::vec4 rot0(
		rotations[4*idx+0],
		rotations[4*idx+1],
		rotations[4*idx+2],
		rotations[4*idx+3]
	);

	// === build Σ₀ ===
	float cov0[6];
	evs_computeCov3D(scale0, 1.0f, rot0, cov0);

	float Sigma0[9];
	unpackCov3D(cov0, Sigma0);

	// === choose split plane ===
	// Use 3 * sigma_max as intersection threshold (99.7% of Gaussian mass)
	float sigma_max = fmaxf(fmaxf(scale0.x, scale0.y), scale0.z);
	float n[3], d;
	bool found = false;
	for (int p = 0; p < n_clips; ++p) {
		n[0] = clippers[4*p+0];
		n[1] = clippers[4*p+1];
		n[2] = clippers[4*p+2];
		d    = clippers[4*p+3];
		float dist = n[0]*mu[0] + n[1]*mu[1] + n[2]*mu[2] + d;
		if (fabsf(dist) <= 3.0f * sigma_max) { found = true; break; }
	}
	if (!found) return;

	// === EVS math ===
	float d0 = n[0]*mu[0] + n[1]*mu[1] + n[2]*mu[2] + d;

	float L0[3] = {
		Sigma0[0]*n[0] + Sigma0[1]*n[1] + Sigma0[2]*n[2],
		Sigma0[3]*n[0] + Sigma0[4]*n[1] + Sigma0[5]*n[2],
		Sigma0[6]*n[0] + Sigma0[7]*n[1] + Sigma0[8]*n[2]
	};

	float tau2 = n[0]*L0[0] + n[1]*L0[1] + n[2]*L0[2];
	float tau  = sqrtf(fmaxf(tau2, 1e-8f));

	float t = d0 / (1.41421356f * tau);
	float Cl = 0.5f * (1.0f - erff(t));
	float Cr = 1.0f - Cl;
	Cl = fmaxf(Cl, 1e-6f);
	Cr = fmaxf(Cr, 1e-6f);

	float D = 0.39894228f * expf(-0.5f * d0 * d0 / (tau*tau));

	// Note: benefit-cost check is now done in preprocess, so we always split here

	// === write output index ===
	uint32_t out_base = 2 * k;
	out_src_indices[k] = idx;

	// === new opacities ===
	float alpha_l = alpha0 * Cl;
	float alpha_r = alpha0 * Cr;

	// === new means ===
	float mu_l[3], mu_r[3];
	float shift_l = D / (tau * Cl);
	float shift_r = D / (tau * Cr);
	for (int i = 0; i < 3; ++i) {
		mu_l[i] = mu[i] - L0[i] * shift_l;
		mu_r[i] = mu[i] + L0[i] * shift_r;
	}

	// === new covariances ===
	float k_l = d0*D/(tau*Cl) - (D*D)/(Cl*Cl);
	float k_r = d0*D/(tau*Cr) + (D*D)/(Cr*Cr);

	float Sigma_l[9], Sigma_r[9];
	for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j) {
		float outer = (L0[i]*L0[j])/(tau*tau);
		Sigma_l[3*i+j] = Sigma0[3*i+j] + outer * k_l;
		Sigma_r[3*i+j] = Sigma0[3*i+j] - outer * k_r;
	}

	// === project back to (scale, quat) ===
	float scale_l[3], scale_r[3];
	float quat_l[4], quat_r[4];
	covToScaleRot(Sigma_l, scale_l, quat_l);
	covToScaleRot(Sigma_r, scale_r, quat_r);

	// === write outputs ===
	for (int i = 0; i < 3; ++i) {
		out_means3D[3*(out_base)+i]     = mu_l[i];
		out_means3D[3*(out_base+1)+i]   = mu_r[i];
		out_scales[3*(out_base)+i]      = scale_l[i];
		out_scales[3*(out_base+1)+i]    = scale_r[i];
	}
	for (int i = 0; i < 4; ++i) {
		out_rotations[4*(out_base)+i]   = quat_l[i];
		out_rotations[4*(out_base+1)+i] = quat_r[i];
	}
	out_opacities[out_base]     = alpha_l;
	out_opacities[out_base + 1] = alpha_r;
}


// DL: 下面的版本是简化的可行的EV splitting的实现 ，作为备份 ，目前未启用
// __global__ void EVS_split_kernel(
//     int P,
//     int num_splits,  // 总共要 split 的数量，用于边界检查
//     const float* means3D,
//     const float* scales,
//     const float* rotations,
//     const float* opacities,

//     const uint32_t* split_flag,
//     const uint32_t* split_offsets,

//     int n_clips,
//     const float* clippers,

//     float* out_means3D,
//     float* out_scales,
//     float* out_rotations,
//     float* out_opacities,
//     uint32_t* out_src_indices
// )
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= P) return;
//     if (!split_flag[idx]) return;

//     // 第 k 个被 split 的 Gaussian
//     uint32_t k = split_offsets[idx];

//     // 添加边界检查，防止越界写入
//     if (k >= (uint32_t)num_splits) return;

//     uint32_t out_base = 2 * k;

//     // 记录原 idx（方便外层用）
//     out_src_indices[k] = idx;

//     // 读原 Gaussian
//     float mu[3] = {
//         means3D[3*idx + 0],
//         means3D[3*idx + 1],
//         means3D[3*idx + 2]
//     };

//     float sigma[3] = {
//         scales[3*idx + 0],
//         scales[3*idx + 1],
//         scales[3*idx + 2]
//     };

//     float q[4] = {
//         rotations[4*idx + 0],
//         rotations[4*idx + 1],
//         rotations[4*idx + 2],
//         rotations[4*idx + 3]
//     };

//     float alpha = opacities[idx];

//     // === 选择一个 plane（第一个命中的 plane） ===
//     // 如果没有 plane 与 Gaussian 相交，则不 split（返回）
//     float n[3] = {0, 0, 0};
//     float d = 0;
//     bool found_plane = false;
//     for (int p = 0; p < n_clips; ++p) {
//         n[0] = clippers[4*p + 0];
//         n[1] = clippers[4*p + 1];
//         n[2] = clippers[4*p + 2];
//         d    = clippers[4*p + 3];

//         float dist = dot3(mu, n) + d;
//         float sigma_max = fmaxf(fmaxf(sigma[0], sigma[1]), sigma[2]);
//         if (fabs(dist) <= 3.0f * sigma_max) {
//             found_plane = true;
//             break;
//         }
//     }
//     // 如果没有找到任何相交的 plane，不进行 split
//     if (!found_plane) return;

//     // === MVP EVSplitting ===
//     float d0 = dot3(mu, n) + d;
//     float tau = fmaxf(1e-6f, fmaxf(fmaxf(sigma[0], sigma[1]), sigma[2]));

//     float inv_sqrt2 = 0.70710678f;
//     float t = d0 / (tau * inv_sqrt2);

//     float Cl = 0.5f * (1.0f - erff(t));
//     float Cr = 1.0f - Cl;

//     Cl = fmaxf(Cl, 1e-6f);
//     Cr = fmaxf(Cr, 1e-6f);

//     // new opacities
//     float alpha_l = alpha * Cl;
//     float alpha_r = alpha * Cr;

//     // mean shift
//     float shift = tau * 0.5f;

//     float mu_l[3] = {
//         mu[0] - shift * n[0],
//         mu[1] - shift * n[1],
//         mu[2] - shift * n[2]
//     };

//     float mu_r[3] = {
//         mu[0] + shift * n[0],
//         mu[1] + shift * n[1],
//         mu[2] + shift * n[2]
//     };

//     // === 写 left ===
//     out_means3D[3*out_base + 0] = mu_l[0];
//     out_means3D[3*out_base + 1] = mu_l[1];
//     out_means3D[3*out_base + 2] = mu_l[2];

//     out_scales[3*out_base + 0] = sigma[0];
//     out_scales[3*out_base + 1] = sigma[1];
//     out_scales[3*out_base + 2] = sigma[2];

//     out_rotations[4*out_base + 0] = q[0];
//     out_rotations[4*out_base + 1] = q[1];
//     out_rotations[4*out_base + 2] = q[2];
//     out_rotations[4*out_base + 3] = q[3];

//     out_opacities[out_base] = alpha_l;

//     // === 写 right ===
//     out_means3D[3*(out_base+1) + 0] = mu_r[0];
//     out_means3D[3*(out_base+1) + 1] = mu_r[1];
//     out_means3D[3*(out_base+1) + 2] = mu_r[2];

//     out_scales[3*(out_base+1) + 0] = sigma[0];
//     out_scales[3*(out_base+1) + 1] = sigma[1];
//     out_scales[3*(out_base+1) + 2] = sigma[2];

//     out_rotations[4*(out_base+1) + 0] = q[0];
//     out_rotations[4*(out_base+1) + 1] = q[1];
//     out_rotations[4*(out_base+1) + 2] = q[2];
//     out_rotations[4*(out_base+1) + 3] = q[3];

//     out_opacities[out_base + 1] = alpha_r;
// }