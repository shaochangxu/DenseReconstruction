// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define _USE_MATH_DEFINES

#include "mvs/patch_match_cuda.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <sstream>

#include "util/cuda.h"
#include "util/cudacc.h"
#include "util/logging.h"

// The number of threads per Cuda thread. Warning: Do not change this value,
// since the templated window sizes rely on this value.
#define THREADS_PER_BLOCK 32

// We must not include "util/math.h" to avoid any Eigen includes here,
// since Visual Studio cannot compile some of the Eigen/Boost expressions.
#ifndef DEG2RAD
#define DEG2RAD(deg) deg * 0.0174532925199432
#endif

namespace colmap {
namespace mvs {

texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat>
    ref_image_texture;
texture<uint8_t, cudaTextureType2DLayered, cudaReadModeNormalizedFloat>
    src_images_texture;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
    src_depth_maps_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType> poses_texture;

// Calibration of reference image as {fx, cx, fy, cy}.
__constant__ float ref_K[4];
// Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
__constant__ float ref_inv_K[4];

__device__ inline void Mat33DotVec3(const float mat[9], const float vec[3],
                                    float result[3]) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ inline void Mat33DotVec3Homogeneous(const float mat[9],
                                               const float vec[2],
                                               float result[2]) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

__device__ inline float DotProduct3(const float vec1[3], const float vec2[3]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

__device__ inline float GenerateRandomDepth(const float depth_min,
                                            const float depth_max,
                                            curandState* rand_state) {
  return curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
}

__device__ inline void GenerateRandomNormal(const int row, const int col,
                                            curandState* rand_state,
                                            float normal[3]) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
    v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  // Make sure normal is looking away from camera.
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3], 1.0f};
  if (DotProduct3(normal, view_ray) > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

__device__ inline float PerturbDepth(const float perturbation,
                                     const float depth,
                                     curandState* rand_state) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return GenerateRandomDepth(depth_min, depth_max, rand_state);
}

__device__ inline void PerturbNormal(const int row, const int col,
                                     const float perturbation,
                                     const float normal[3],
                                     curandState* rand_state,
                                     float perturbed_normal[3],
                                     const int num_trials = 0) {
  // Perturbation rotation angles.
  const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

  const float sin_a1 = sin(a1);
  const float sin_a2 = sin(a2);
  const float sin_a3 = sin(a3);
  const float cos_a1 = cos(a1);
  const float cos_a2 = cos(a2);
  const float cos_a3 = cos(a3);

  // R = Rx * Ry * Rz
  float R[9];
  R[0] = cos_a2 * cos_a3;
  R[1] = -cos_a2 * sin_a3;
  R[2] = sin_a2;
  R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
  R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
  R[5] = -cos_a2 * sin_a1;
  R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
  R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
  R[8] = cos_a1 * cos_a2;

  // Perturb the normal vector.
  Mat33DotVec3(R, normal, perturbed_normal);

  // Make sure the perturbed normal is still looking in the same direction as
  // the viewing direction, otherwise try again but with smaller perturbation.
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3], 1.0f};
  if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
    const int kMaxNumTrials = 3;
    if (num_trials < kMaxNumTrials) {
      PerturbNormal(row, col, 0.5f * perturbation, normal, rand_state,
                    perturbed_normal, num_trials + 1);
      return;
    } else {
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }
  }

  // Make sure normal has unit norm.
  const float inv_norm = rsqrt(DotProduct3(perturbed_normal, perturbed_normal));
  perturbed_normal[0] *= inv_norm;
  perturbed_normal[1] *= inv_norm;
  perturbed_normal[2] *= inv_norm;
}

__device__ inline void ComputePointAtDepth(const float row, const float col,
                                           const float depth, float point[3]) {
  point[0] = depth * (ref_inv_K[0] * col + ref_inv_K[1]);
  point[1] = depth * (ref_inv_K[2] * row + ref_inv_K[3]);
  point[2] = depth;
}

// Transfer depth on plane from viewing ray at row1 to row2. The returned
// depth is the intersection of the viewing ray through row2 with the plane
// at row1 defined by the given depth and normal.
__device__ inline float PropagateDepth(const float depth1,
                                       const float normal1[3], const float row1,
                                       const float row2) {
  // Point along first viewing ray.
  const float x1 = depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]);
  const float y1 = depth1;
  // Point on plane defined by point along first viewing ray and plane normal1.
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];

  // Origin of second viewing ray.
  // const float x3 = 0.0f;
  // const float y3 = 0.0f;
  // Point on second viewing ray.
  const float x4 = ref_inv_K[2] * row2 + ref_inv_K[3];
  // const float y4 = 1.0f;

  // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
  const float denom = x2 - x1 + x4 * (y1 - y2);
  constexpr float kEps = 1e-5f;
  if (abs(denom) < kEps) {
    return depth1;
  }
  const float nom = y1 * x2 - x1 * y2;
  return nom / denom;
}

// First, compute triangulation angle between reference and source image for 3D
// point. Second, compute incident angle between viewing direction of source
// image and normal direction of 3D point. Both angles are cosine distances.
__device__ inline void ComputeViewingAngles(const float point[3],
                                            const float normal[3],
                                            const int image_idx,
                                            float* cos_triangulation_angle,
                                            float* cos_incident_angle) {
  *cos_triangulation_angle = 0.0f;
  *cos_incident_angle = 0.0f;

  // Projection center of source image.
  float C[3];
  for (int i = 0; i < 3; ++i) {
    C[i] = tex2D(poses_texture, i + 16, image_idx);
  }

  // Ray from point to camera.
  const float SX[3] = {C[0] - point[0], C[1] - point[1], C[2] - point[2]};

  // Length of ray from reference image to point.
  const float RX_inv_norm = rsqrt(DotProduct3(point, point));

  // Length of ray from source image to point.
  const float SX_inv_norm = rsqrt(DotProduct3(SX, SX));

  *cos_incident_angle = DotProduct3(SX, normal) * SX_inv_norm;
  *cos_triangulation_angle = DotProduct3(SX, point) * RX_inv_norm * SX_inv_norm;
}

__device__ inline void ComposeHomography(const int image_idx, const int row,
                                         const int col, const float depth,
                                         const float normal[3], float H[9]) {
  // Calibration of source image.
  float K[4];
  for (int i = 0; i < 4; ++i) {
    K[i] = tex2D(poses_texture, i, image_idx);
  }

  // Relative rotation between reference and source image.
  float R[9];
  for (int i = 0; i < 9; ++i) {
    R[i] = tex2D(poses_texture, i + 4, image_idx);
  }

  // Relative translation between reference and source image.
  float T[3];
  for (int i = 0; i < 3; ++i) {
    T[i] = tex2D(poses_texture, i + 13, image_idx);
  }

  // Distance to the plane.
  const float dist =
      depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
               normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;

  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  // Homography as H = K * (R - T * n' / d) * Kref^-1.
  H[0] = ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                         K[1] * (R[6] + inv_dist_N0 * T[2]));
  H[1] = ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                         K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
         K[1] * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                         K[1] * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                         K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[3] = ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                         K[3] * (R[6] + inv_dist_N0 * T[2]));
  H[4] = ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                         K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
         K[3] * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                         K[3] * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                         K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
  H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
  H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
         ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
}

// Each thread in the current warp / thread block reads in 3 columns of the
// reference image. The shared memory holds 3 * THREADS_PER_BLOCK columns and
// kWindowSize rows of the reference image. Each thread copies every
// THREADS_PER_BLOCK-th column from global to shared memory offset by its ID.
// For example, if THREADS_PER_BLOCK = 32, then thread 0 reads columns 0, 32, 64
// and thread 1 columns 1, 33, 65. When computing the photoconsistency, which is
// shared among each thread block, each thread can then read the reference image
// colors from shared memory. Note that this limits the window radius to a
// maximum of THREADS_PER_BLOCK.
template <int kWindowSize>
struct LocalRefImage {
  const static int kWindowRadius = kWindowSize / 2;
  const static int kThreadBlockRadius = 1;
  const static int kThreadBlockSize = 2 * kThreadBlockRadius + 1;
  const static int kNumRows = kWindowSize;
  const static int kNumColumns = kThreadBlockSize * THREADS_PER_BLOCK;
  const static int kDataSize = kNumRows * kNumColumns;

  float* data = nullptr;

  __device__ inline void Read(const int row) {
    // For the first row, read the entire block into shared memory. For all
    // consecutive rows, it is only necessary to shift the rows in shared memory
    // up by one element and then read in a new row at the bottom of the shared
    // memory. Note that this assumes that the calling loop starts with the
    // first row and then consecutively reads in the next row.

    const int thread_id = threadIdx.x;
    const int thread_block_first_id = blockDim.x * blockIdx.x;

    const int local_col_start = thread_id;
    const int global_col_start = thread_block_first_id -
                                 kThreadBlockRadius * THREADS_PER_BLOCK +
                                 thread_id;

    if (row == 0) {
      int global_row = row - kWindowRadius;
      for (int local_row = 0; local_row < kNumRows; ++local_row, ++global_row) {
        int local_col = local_col_start;
        int global_col = global_col_start;
#pragma unroll
        for (int block = 0; block < kThreadBlockSize; ++block) {
          data[local_row * kNumColumns + local_col] =
              tex2D(ref_image_texture, global_col, global_row);
          local_col += THREADS_PER_BLOCK;
          global_col += THREADS_PER_BLOCK;
        }
      }
    } else {
      // Move rows in shared memory up by one row.
      for (int local_row = 1; local_row < kNumRows; ++local_row) {
        int local_col = local_col_start;
#pragma unroll
        for (int block = 0; block < kThreadBlockSize; ++block) {
          data[(local_row - 1) * kNumColumns + local_col] =
              data[local_row * kNumColumns + local_col];
          local_col += THREADS_PER_BLOCK;
        }
      }

      // Read next row into the last row of shared memory.
      const int local_row = kNumRows - 1;
      const int global_row = row + kWindowRadius;
      int local_col = local_col_start;
      int global_col = global_col_start;
#pragma unroll
      for (int block = 0; block < kThreadBlockSize; ++block) {
        data[local_row * kNumColumns + local_col] =
            tex2D(ref_image_texture, global_col, global_row);
        local_col += THREADS_PER_BLOCK;
        global_col += THREADS_PER_BLOCK;
      }
    }
  }
};

// The return values is 1 - NCC, so the range is [0, 2], the smaller the
// value, the better the color consistency.
template <int kWindowSize, int kWindowStep>
struct PhotoConsistencyCostComputer {
  const static int kWindowRadius = kWindowSize / 2;

  __device__ PhotoConsistencyCostComputer(const float sigma_spatial,
                                          const float sigma_color)
      : bilateral_weight_computer_(sigma_spatial, sigma_color) {}

  // Maximum photo consistency cost as 1 - min(NCC).
  const float kMaxCost = 2.0f;

  // Thread warp local reference image data around current patch.
  typedef LocalRefImage<kWindowSize> LocalRefImageType;
  LocalRefImageType local_ref_image;

  // Precomputed sum of raw and squared image intensities.
  float local_ref_sum = 0.0f;
  float local_ref_squared_sum = 0.0f;

  // Index of source image.
  int src_image_idx = -1;

  // Center position of patch in reference image.
  int row = -1;
  int col = -1;

  // Depth and normal for which to warp patch.
  float depth = 0.0f;
  const float* normal = nullptr;

  __device__ inline void Read(const int row) {
    local_ref_image.Read(row);
    __syncthreads();
  }

  __device__ inline float Compute() const {
    float tform[9];
    ComposeHomography(src_image_idx, row, col, depth, normal, tform);

    float tform_step[8];
    for (int i = 0; i < 8; ++i) {
      tform_step[i] = kWindowStep * tform[i];
    }

    const int thread_id = threadIdx.x;
    const int row_start = row - kWindowRadius;
    const int col_start = col - kWindowRadius;

    float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
    float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
    float z = tform[6] * col_start + tform[7] * row_start + tform[8];
    float base_col_src = col_src;
    float base_row_src = row_src;
    float base_z = z;

    int ref_image_idx = THREADS_PER_BLOCK - kWindowRadius + thread_id;
    int ref_image_base_idx = ref_image_idx;

    const float ref_center_color =
        local_ref_image
            .data[ref_image_idx + kWindowRadius * 3 * THREADS_PER_BLOCK +
                  kWindowRadius];
    const float ref_color_sum = local_ref_sum;
    const float ref_color_squared_sum = local_ref_squared_sum;
    float src_color_sum = 0.0f;
    float src_color_squared_sum = 0.0f;
    float src_ref_color_sum = 0.0f;
    float bilateral_weight_sum = 0.0f;

    for (int row = -kWindowRadius; row <= kWindowRadius; row += kWindowStep) {
      for (int col = -kWindowRadius; col <= kWindowRadius; col += kWindowStep) {
        const float inv_z = 1.0f / z;
        const float norm_col_src = inv_z * col_src + 0.5f;
        const float norm_row_src = inv_z * row_src + 0.5f;
        const float ref_color = local_ref_image.data[ref_image_idx];
        const float src_color = tex2DLayered(src_images_texture, norm_col_src,
                                             norm_row_src, src_image_idx);

        const float bilateral_weight = bilateral_weight_computer_.Compute(
            row, col, ref_center_color, ref_color);

        const float bilateral_weight_src = bilateral_weight * src_color;

        src_color_sum += bilateral_weight_src;
        src_color_squared_sum += bilateral_weight_src * src_color;
        src_ref_color_sum += bilateral_weight_src * ref_color;
        bilateral_weight_sum += bilateral_weight;

        ref_image_idx += kWindowStep;

        // Accumulate warped source coordinates per row to reduce numerical
        // errors. Note that this is necessary since coordinates usually are in
        // the order of 1000s as opposed to the color values which are
        // normalized to the range [0, 1].
        col_src += tform_step[0];
        row_src += tform_step[3];
        z += tform_step[6];
      }

      ref_image_base_idx += kWindowStep * 3 * THREADS_PER_BLOCK;
      ref_image_idx = ref_image_base_idx;

      base_col_src += tform_step[1];
      base_row_src += tform_step[4];
      base_z += tform_step[7];

      col_src = base_col_src;
      row_src = base_row_src;
      z = base_z;
    }

    const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
    src_color_sum *= inv_bilateral_weight_sum;
    src_color_squared_sum *= inv_bilateral_weight_sum;
    src_ref_color_sum *= inv_bilateral_weight_sum;

    const float ref_color_var =
        ref_color_squared_sum - ref_color_sum * ref_color_sum;
    const float src_color_var =
        src_color_squared_sum - src_color_sum * src_color_sum;

    // Based on Jensen's Inequality for convex functions, the variance
    // should always be larger than 0. Do not make this threshold smaller.
    constexpr float kMinVar = 1e-5f;
    if (ref_color_var < kMinVar || src_color_var < kMinVar) {
      return kMaxCost;
    } else {
      const float src_ref_color_covar =
          src_ref_color_sum - ref_color_sum * src_color_sum;
      const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
      return max(0.0f,
                 min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
    }
  }

  __device__ inline float ACMMCompute_shared(const int thread_idx_x, const int thread_idx_y) const {
  	

    float tform[9];
    ComposeHomography(src_image_idx, row, col, depth, normal, tform);

    float tform_step[9];
    for (int i = 0; i < 9; ++i) {
      tform_step[i] = kWindowStep * tform[i];
    }

    const int row_start = row - kWindowRadius;
    const int col_start = col - kWindowRadius;

    float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
    float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
    float z = tform[6] * col_start + tform[7] * row_start + tform[8];
    float base_col_src = col_src;
    float base_row_src = row_src;
    float base_z = z;

    // shared memory
 //    const int shared_width = 3 * THREADS_PER_BLOCK;
 //    const int shared_idx = THREADS_PER_BLOCK + thread_idx_x;
	// const int shared_idy = THREADS_PER_BLOCK + thread_idx_y;
	// const int cur_idx = shared_idy * shared_width + shared_idx;
 //    int ref_image_idx = cur_idx - kWindowRadius * shared_width- kWindowRadius;
 //    int ref_image_base_idx = ref_image_idx;
    const int shared_width = (THREADS_PER_BLOCK + 2 * kWindowRadius);
    const int cur_idx = (thread_idx_y + kWindowRadius) * shared_width+ kWindowRadius + thread_idx_x;

    int ref_image_idx = cur_idx - kWindowRadius * shared_width- kWindowRadius;
    int ref_image_base_idx = ref_image_idx;

    const float ref_center_color = local_ref_image[cur_idx];
    
    const float ref_color_sum = local_ref_sum;
    const float ref_color_squared_sum = local_ref_squared_sum;
    float src_color_sum = 0.0f;
    float src_color_squared_sum = 0.0f;
    float src_ref_color_sum = 0.0f;
    float bilateral_weight_sum = 0.0f;

    //int index = 0;
    for (int r = -kWindowRadius; r <= kWindowRadius; r += kWindowStep) {
      for (int c = -kWindowRadius; c <= kWindowRadius; c += kWindowStep) {
       const float inv_z = 1.0f / z;
        const float norm_col_src = inv_z * col_src + 0.5f;
        const float norm_row_src = inv_z * row_src + 0.5f;
        
        const float ref_color = local_ref_image[ref_image_idx];//tex2D(ref_image_texture, col + c, row + r);//
        const float src_color = tex2DLayered(src_images_texture, norm_col_src,
                                             norm_row_src, src_image_idx);

        const float bilateral_weight = bilateral_weight_computer_.Compute(
            r, c, ref_center_color, ref_color);

        const float bilateral_weight_src = bilateral_weight * src_color;

        src_color_sum += bilateral_weight_src;
        src_color_squared_sum += bilateral_weight_src * src_color;
        src_ref_color_sum += bilateral_weight_src * ref_color;
        bilateral_weight_sum += bilateral_weight;

        ref_image_idx += kWindowStep;

        // Accumulate warped source coordinates per row to reduce numerical
        // errors. Note that this is necessary since coordinates usually are in
        // the order of 1000s as opposed to the color values which are
        // normalized to the range [0, 1].
        col_src += tform_step[0];
        row_src += tform_step[3];
        z += tform_step[6];
      	
 
      }

      ref_image_base_idx += kWindowStep * shared_width;
      ref_image_idx = ref_image_base_idx;

      base_col_src += tform_step[1];
      base_row_src += tform_step[4];
      base_z += tform_step[7];

      col_src = base_col_src;
      row_src = base_row_src;
      z = base_z;
    }

    const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
    src_color_sum *= inv_bilateral_weight_sum;
    src_color_squared_sum *= inv_bilateral_weight_sum;
    src_ref_color_sum *= inv_bilateral_weight_sum;

    const float ref_color_var =
        ref_color_squared_sum - ref_color_sum * ref_color_sum;
    const float src_color_var =
        src_color_squared_sum - src_color_sum * src_color_sum;

    // Based on Jensen's Inequality for convex functions, the variance
    // should always be larger than 0. Do not make this threshold smaller.
    const float kMinVar = 1e-5f;
    //printf("%f\n", ref_center_color);
    if (ref_color_var < kMinVar || src_color_var < kMinVar) {
      return kMaxCost;
    } else {
      const float src_ref_color_covar =
          src_ref_color_sum - ref_color_sum * src_color_sum;
      const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
      return max(0.0f,
                 min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
    }
    //return 0.7f;
  }

 private:
  const BilateralWeightComputer bilateral_weight_computer_;
};

__device__ inline float ComputeGeomConsistencyCost(const float row,
                                                   const float col,
                                                   const float depth,
                                                   const int image_idx,
                                                   const float max_cost) {
  // Extract projection matrices for source image.
  float P[12];
  for (int i = 0; i < 12; ++i) {
    P[i] = tex2D(poses_texture, i + 19, image_idx);
  }
  float inv_P[12];
  for (int i = 0; i < 12; ++i) {
    inv_P[i] = tex2D(poses_texture, i + 31, image_idx);
  }

  // Project point in reference image to world.
  float forward_point[3];
  ComputePointAtDepth(row, col, depth, forward_point);

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = tex2DLayered(src_depth_maps_texture, src_col + 0.5f,
                                       src_row + 0.5f, image_idx);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col =
      inv_backward_point_z *
      (ref_K[0] * backward_point_x + ref_K[1] * backward_point_z);
  const float backward_row =
      inv_backward_point_z *
      (ref_K[2] * backward_point_y + ref_K[3] * backward_point_z);

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

// Find index of minimum in given values.
template <int kNumCosts>
__device__ inline int FindMinCost(const float costs[kNumCosts]) {
  float min_cost = costs[0];
  int min_cost_idx = 0;
  for (int idx = 1; idx < kNumCosts; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

__device__ inline void TransformPDFToCDF(float* probs, const int num_probs) {
  float prob_sum = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    prob_sum += probs[i];
  }
  const float inv_prob_sum = 1.0f / prob_sum;

  float cum_prob = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    const float prob = probs[i] * inv_prob_sum;
    cum_prob += prob;
    probs[i] = cum_prob;
  }
}

class LikelihoodComputer {
 public:
  __device__ LikelihoodComputer(const float ncc_sigma,
                                const float min_triangulation_angle,
                                const float incident_angle_sigma)
      : cos_min_triangulation_angle_(cos(min_triangulation_angle)),
        inv_incident_angle_sigma_square_(
            -0.5f / (incident_angle_sigma * incident_angle_sigma)),
        inv_ncc_sigma_square_(-0.5f / (ncc_sigma * ncc_sigma)),
        ncc_norm_factor_(ComputeNCCCostNormFactor(ncc_sigma)) {}

  // Compute forward message from current cost and forward message of
  // previous / neighboring pixel.
  __device__ float ComputeForwardMessage(const float cost,
                                         const float prev) const {
    return ComputeMessage<true>(cost, prev);
  }

  // Compute backward message from current cost and backward message of
  // previous / neighboring pixel.
  __device__ float ComputeBackwardMessage(const float cost,
                                          const float prev) const {
    return ComputeMessage<false>(cost, prev);
  }

  // Compute the selection probability from the forward and backward message.
  __device__ inline float ComputeSelProb(const float alpha, const float beta,
                                         const float prev,
                                         const float prev_weight) const {
    const float zn0 = (1.0f - alpha) * (1.0f - beta);
    const float zn1 = alpha * beta;
    const float curr = zn1 / (zn0 + zn1);
    return prev_weight * prev + (1.0f - prev_weight) * curr;
  }

  // Compute NCC probability. Note that cost = 1 - NCC.
  __device__ inline float ComputeNCCProb(const float cost) const {
    return exp(cost * cost * inv_ncc_sigma_square_) * ncc_norm_factor_;
  }

  // Compute the triangulation angle probability.
  __device__ inline float ComputeTriProb(
      const float cos_triangulation_angle) const {
    const float abs_cos_triangulation_angle = abs(cos_triangulation_angle);
    if (abs_cos_triangulation_angle > cos_min_triangulation_angle_) {
      const float scaled = 1.0f - (1.0f - abs_cos_triangulation_angle) /
                                      (1.0f - cos_min_triangulation_angle_);
      const float likelihood = 1.0f - scaled * scaled;
      return min(1.0f, max(0.0f, likelihood));
    } else {
      return 1.0f;
    }
  }

  // Compute the incident angle probability.
  __device__ inline float ComputeIncProb(const float cos_incident_angle) const {
    const float x = 1.0f - max(0.0f, cos_incident_angle);
    return exp(x * x * inv_incident_angle_sigma_square_);
  }

  // Compute the warping/resolution prior probability.
  template <int kWindowSize>
  __device__ inline float ComputeResolutionProb(const float H[9],
                                                const float row,
                                                const float col) const {
    const int kWindowRadius = kWindowSize / 2;

    // Warp corners of patch in reference image to source image.
    float src1[2];
    const float ref1[2] = {col - kWindowRadius, row - kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref1, src1);
    float src2[2];
    const float ref2[2] = {col - kWindowRadius, row + kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref2, src2);
    float src3[2];
    const float ref3[2] = {col + kWindowRadius, row + kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref3, src3);
    float src4[2];
    const float ref4[2] = {col + kWindowRadius, row - kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref4, src4);

    // Compute area of patches in reference and source image.
    const float ref_area = kWindowSize * kWindowSize;
    const float src_area =
        abs(0.5f * (src1[0] * src2[1] - src2[0] * src1[1] - src1[0] * src4[1] +
                    src2[0] * src3[1] - src3[0] * src2[1] + src4[0] * src1[1] +
                    src3[0] * src4[1] - src4[0] * src3[1]));

    if (ref_area > src_area) {
      return src_area / ref_area;
    } else {
      return ref_area / src_area;
    }
  }

 private:
  // The normalization for the likelihood function, i.e. the normalization for
  // the prior on the matching cost.
  __device__ static inline float ComputeNCCCostNormFactor(
      const float ncc_sigma) {
    // A = sqrt(2pi)*sigma/2*erf(sqrt(2)/sigma)
    // erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
    return 2.0f / (sqrt(2.0f * M_PI) * ncc_sigma *
                   erff(2.0f / (ncc_sigma * 1.414213562f)));
  }

  // Compute the forward or backward message.
  template <bool kForward>
  __device__ inline float ComputeMessage(const float cost,
                                         const float prev) const {
    constexpr float kUniformProb = 0.5f;
    constexpr float kNoChangeProb = 0.99999f;
    const float kChangeProb = 1.0f - kNoChangeProb;
    const float emission = ComputeNCCProb(cost);

    float zn0;  // Message for selection probability = 0.
    float zn1;  // Message for selection probability = 1.
    if (kForward) {
      zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb) * kUniformProb;
      zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb) * emission;
    } else {
      zn0 = prev * emission * kChangeProb +
            (1.0f - prev) * kUniformProb * kNoChangeProb;
      zn1 = prev * emission * kNoChangeProb +
            (1.0f - prev) * kUniformProb * kChangeProb;
    }

    return zn1 / (zn0 + zn1);
  }

  float cos_min_triangulation_angle_;
  float inv_incident_angle_sigma_square_;
  float inv_ncc_sigma_square_;
  float ncc_norm_factor_;
};

// Rotate normals by 90deg around z-axis in counter-clockwise direction.
__global__ void InitNormalMap(GpuMat<float> normal_map,
                              GpuMat<curandState> rand_state_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
    curandState rand_state = rand_state_map.Get(row, col);
    float normal[3];
    GenerateRandomNormal(row, col, &rand_state, normal);
    normal_map.SetSlice(row, col, normal);
    rand_state_map.Set(row, col, rand_state);
  }
}

// Rotate normals by 90deg around z-axis in counter-clockwise direction.
__global__ void RotateNormalMap(GpuMat<float> normal_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
    float normal[3];
    normal_map.GetSlice(row, col, normal);
    float rotated_normal[3];
    rotated_normal[0] = normal[1];
    rotated_normal[1] = -normal[0];
    rotated_normal[2] = normal[2];
    normal_map.SetSlice(row, col, rotated_normal);
  }
}

template <int kWindowSize, int kWindowStep>
__global__ void ComputeInitialCost(GpuMat<float> cost_map,
                                   const GpuMat<float> depth_map,
                                   const GpuMat<float> normal_map,
                                   const GpuMat<float> ref_sum_image,
                                   const GpuMat<float> ref_squared_sum_image,
                                   const float sigma_spatial,
                                   const float sigma_color) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  typedef PhotoConsistencyCostComputer<kWindowSize, kWindowStep>
      PhotoConsistencyCostComputerType;
  PhotoConsistencyCostComputerType pcc_computer(sigma_spatial, sigma_color);
  pcc_computer.col = col;

  __shared__ float local_ref_image_data
      [PhotoConsistencyCostComputerType::LocalRefImageType::kDataSize];
  pcc_computer.local_ref_image.data = &local_ref_image_data[0];

  float normal[3] = {0};
  pcc_computer.normal = normal;

  for (int row = 0; row < cost_map.GetHeight(); ++row) {
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    pcc_computer.Read(row);

    if (col < cost_map.GetWidth()) {
      pcc_computer.depth = depth_map.Get(row, col);
      normal_map.GetSlice(row, col, normal);

      pcc_computer.row = row;
      pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
      pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

      for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
        pcc_computer.src_image_idx = image_idx;
        cost_map.Set(row, col, image_idx, pcc_computer.Compute());
      }
    }
  }
}

struct SweepOptions {
  float perturbation = 1.0f;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  int num_samples = 15;
  float sigma_spatial = 3.0f;
  float sigma_color = 0.3f;
  float ncc_sigma = 0.6f;
  float min_triangulation_angle = 0.5f;
  float incident_angle_sigma = 0.9f;
  float prev_sel_prob_weight = 0.0f;
  float geom_consistency_regularizer = 0.1f;
  float geom_consistency_max_cost = 5.0f;
  float filter_min_ncc = 0.1f;
  float filter_min_triangulation_angle = 3.0f;
  int filter_min_num_consistent = 2;
  float filter_geom_consistency_max_cost = 1.0f;
};

template <int kWindowSize, int kWindowStep, bool kGeomConsistencyTerm = false,
          bool kFilterPhotoConsistency = false,
          bool kFilterGeomConsistency = false>
__global__ void SweepFromTopToBottom(
    GpuMat<float> global_workspace, GpuMat<curandState> rand_state_map,
    GpuMat<float> cost_map, GpuMat<float> depth_map, GpuMat<float> normal_map,
    GpuMat<uint8_t> consistency_mask, GpuMat<float> sel_prob_map,
    const GpuMat<float> prev_sel_prob_map, const GpuMat<float> ref_sum_image,
    const GpuMat<float> ref_squared_sum_image, const SweepOptions options) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  // Probability for boundary pixels.
  constexpr float kUniformProb = 0.5f;

  LikelihoodComputer likelihood_computer(options.ncc_sigma,
                                         options.min_triangulation_angle,
                                         options.incident_angle_sigma);

  float* forward_message =
      &global_workspace.GetPtr()[col * global_workspace.GetHeight()];
  float* sampling_probs =
      &global_workspace.GetPtr()[global_workspace.GetWidth() *
                                     global_workspace.GetHeight() +
                                 col * global_workspace.GetHeight()];

  //////////////////////////////////////////////////////////////////////////////
  // Compute backward message for all rows. Note that the backward messages are
  // temporarily stored in the sel_prob_map and replaced row by row as the
  // updated forward messages are computed further below.
  //////////////////////////////////////////////////////////////////////////////

  if (col < cost_map.GetWidth()) {
    for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
      // Compute backward message.
      float beta = kUniformProb;
      for (int row = cost_map.GetHeight() - 1; row >= 0; --row) {
        const float cost = cost_map.Get(row, col, image_idx);
        beta = likelihood_computer.ComputeBackwardMessage(cost, beta);
        sel_prob_map.Set(row, col, image_idx, beta);
      }

      // Initialize forward message.
      forward_message[image_idx] = kUniformProb;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Estimate parameters for remaining rows and compute selection probabilities.
  //////////////////////////////////////////////////////////////////////////////

  typedef PhotoConsistencyCostComputer<kWindowSize, kWindowStep>
      PhotoConsistencyCostComputerType;
  PhotoConsistencyCostComputerType pcc_computer(options.sigma_spatial,
                                                options.sigma_color);
  pcc_computer.col = col;

  __shared__ float local_ref_image_data
      [PhotoConsistencyCostComputerType::LocalRefImageType::kDataSize];
  pcc_computer.local_ref_image.data = &local_ref_image_data[0];

  struct ParamState {
    float depth = 0.0f;
    float normal[3] = {0};
  };

  // Parameters of previous pixel in column.
  ParamState prev_param_state;
  // Parameters of current pixel in column.
  ParamState curr_param_state;
  // Randomly sampled parameters.
  ParamState rand_param_state;
  // Cuda PRNG state for random sampling.
  curandState rand_state;

  if (col < cost_map.GetWidth()) {
    // Read random state for current column.
    rand_state = rand_state_map.Get(0, col);
    // Parameters for first row in column.
    prev_param_state.depth = depth_map.Get(0, col);
    normal_map.GetSlice(0, col, prev_param_state.normal);
  }

  for (int row = 0; row < cost_map.GetHeight(); ++row) {
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    pcc_computer.Read(row);

    if (col >= cost_map.GetWidth()) {
      continue;
    }

    pcc_computer.row = row;
    pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
    pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

    // Propagate the depth at which the current ray intersects with the plane
    // of the normal of the previous ray. This helps to better estimate
    // the depth of very oblique structures, i.e. pixels whose normal direction
    // is significantly different from their viewing direction.
    prev_param_state.depth = PropagateDepth(
        prev_param_state.depth, prev_param_state.normal, row - 1, row);

    // Read parameters for current pixel from previous sweep.
    curr_param_state.depth = depth_map.Get(row, col);
    normal_map.GetSlice(row, col, curr_param_state.normal);

    // Generate random parameters.
    rand_param_state.depth =
        PerturbDepth(options.perturbation, curr_param_state.depth, &rand_state);
    PerturbNormal(row, col, options.perturbation * M_PI,
                  curr_param_state.normal, &rand_state,
                  rand_param_state.normal);

    // Read in the backward message, compute selection probabilities and
    // modulate selection probabilities with priors.

    float point[3];
    ComputePointAtDepth(row, col, curr_param_state.depth, point);

    for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
      const float cost = cost_map.Get(row, col, image_idx);
      const float alpha = likelihood_computer.ComputeForwardMessage(
          cost, forward_message[image_idx]);
      const float beta = sel_prob_map.Get(row, col, image_idx);
      const float prev_prob = prev_sel_prob_map.Get(row, col, image_idx);
      const float sel_prob = likelihood_computer.ComputeSelProb(
          alpha, beta, prev_prob, options.prev_sel_prob_weight);

      float cos_triangulation_angle;
      float cos_incident_angle;
      ComputeViewingAngles(point, curr_param_state.normal, image_idx,
                           &cos_triangulation_angle, &cos_incident_angle);
      const float tri_prob =
          likelihood_computer.ComputeTriProb(cos_triangulation_angle);
      const float inc_prob =
          likelihood_computer.ComputeIncProb(cos_incident_angle);

      float H[9];
      ComposeHomography(image_idx, row, col, curr_param_state.depth,
                        curr_param_state.normal, H);
      const float res_prob =
          likelihood_computer.ComputeResolutionProb<kWindowSize>(H, row, col);

      sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
    }

    TransformPDFToCDF(sampling_probs, cost_map.GetDepth());

    // Compute matching cost using Monte Carlo sampling of source images. Images
    // with higher selection probability are more likely to be sampled. Hence,
    // if only very few source images see the reference image pixel, the same
    // source image is likely to be sampled many times. Instead of taking
    // the best K probabilities, this sampling scheme has the advantage of
    // being adaptive to any distribution of selection probabilities.

    constexpr int kNumCosts = 5;
    float costs[kNumCosts] = {0};
    const float depths[kNumCosts] = {
        curr_param_state.depth, prev_param_state.depth, rand_param_state.depth,
        curr_param_state.depth, rand_param_state.depth};
    const float* normals[kNumCosts] = {
        curr_param_state.normal, prev_param_state.normal,
        rand_param_state.normal, rand_param_state.normal,
        curr_param_state.normal};

    for (int sample = 0; sample < options.num_samples; ++sample) {
      const float rand_prob = curand_uniform(&rand_state) - FLT_EPSILON;

      pcc_computer.src_image_idx = -1;
      for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
        const float prob = sampling_probs[image_idx];
        if (prob > rand_prob) {
          pcc_computer.src_image_idx = image_idx;
          break;
        }
      }

      if (pcc_computer.src_image_idx == -1) {
        continue;
      }

      costs[0] += cost_map.Get(row, col, pcc_computer.src_image_idx);
      if (kGeomConsistencyTerm) {
        costs[0] += options.geom_consistency_regularizer *
                    ComputeGeomConsistencyCost(
                        row, col, depths[0], pcc_computer.src_image_idx,
                        options.geom_consistency_max_cost);
      }

      for (int i = 1; i < kNumCosts; ++i) {
        pcc_computer.depth = depths[i];
        pcc_computer.normal = normals[i];
        costs[i] += pcc_computer.Compute();
        if (kGeomConsistencyTerm) {
          costs[i] += options.geom_consistency_regularizer *
                      ComputeGeomConsistencyCost(
                          row, col, depths[i], pcc_computer.src_image_idx,
                          options.geom_consistency_max_cost);
        }
      }
    }

    // Find the parameters of the minimum cost.
    const int min_cost_idx = FindMinCost<kNumCosts>(costs);
    const float best_depth = depths[min_cost_idx];
    const float* best_normal = normals[min_cost_idx];

    // Save best new parameters.
    depth_map.Set(row, col, best_depth);
    normal_map.SetSlice(row, col, best_normal);

    // Use the new cost to recompute the updated forward message and
    // the selection probability.
    pcc_computer.depth = best_depth;
    pcc_computer.normal = best_normal;
    for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
      // Determine the cost for best depth.
      float cost;
      if (min_cost_idx == 0) {
        cost = cost_map.Get(row, col, image_idx);
      } else {
        pcc_computer.src_image_idx = image_idx;
        cost = pcc_computer.Compute();
        cost_map.Set(row, col, image_idx, cost);
      }

      const float alpha = likelihood_computer.ComputeForwardMessage(
          cost, forward_message[image_idx]);
      const float beta = sel_prob_map.Get(row, col, image_idx);
      const float prev_prob = prev_sel_prob_map.Get(row, col, image_idx);
      const float prob = likelihood_computer.ComputeSelProb(
          alpha, beta, prev_prob, options.prev_sel_prob_weight);
      forward_message[image_idx] = alpha;
      sel_prob_map.Set(row, col, image_idx, prob);
    }

    if (kFilterPhotoConsistency || kFilterGeomConsistency) {
      int num_consistent = 0;

      float best_point[3];
      ComputePointAtDepth(row, col, best_depth, best_point);

      const float min_ncc_prob =
          likelihood_computer.ComputeNCCProb(1.0f - options.filter_min_ncc);
      const float cos_min_triangulation_angle =
          cos(options.filter_min_triangulation_angle);

      for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
        float cos_triangulation_angle;
        float cos_incident_angle;
        ComputeViewingAngles(best_point, best_normal, image_idx,
                             &cos_triangulation_angle, &cos_incident_angle);
        if (cos_triangulation_angle > cos_min_triangulation_angle ||
            cos_incident_angle <= 0.0f) {
          continue;
        }

        if (!kFilterGeomConsistency) {
          if (sel_prob_map.Get(row, col, image_idx) >= min_ncc_prob) {
            consistency_mask.Set(row, col, image_idx, 1);
            num_consistent += 1;
          }
        } else if (!kFilterPhotoConsistency) {
          if (ComputeGeomConsistencyCost(row, col, best_depth, image_idx,
                                         options.geom_consistency_max_cost) <=
              options.filter_geom_consistency_max_cost) {
            consistency_mask.Set(row, col, image_idx, 1);
            num_consistent += 1;
          }
        } else {
          if (sel_prob_map.Get(row, col, image_idx) >= min_ncc_prob &&
              ComputeGeomConsistencyCost(row, col, best_depth, image_idx,
                                         options.geom_consistency_max_cost) <=
                  options.filter_geom_consistency_max_cost) {
            consistency_mask.Set(row, col, image_idx, 1);
            num_consistent += 1;
          }
        }
      }

      if (num_consistent < options.filter_min_num_consistent) {
        depth_map.Set(row, col, 0.0f);
        normal_map.Set(row, col, 0, 0.0f);
        normal_map.Set(row, col, 1, 0.0f);
        normal_map.Set(row, col, 2, 0.0f);
        for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
          consistency_mask.Set(row, col, image_idx, 0);
        }
      }
    }

    // Update previous depth for next row.
    prev_param_state.depth = best_depth;
    for (int i = 0; i < 3; ++i) {
      prev_param_state.normal[i] = best_normal[i];
    }
  }

  if (col < cost_map.GetWidth()) {
    rand_state_map.Set(0, col, rand_state);
  }
}

PatchMatchCuda::PatchMatchCuda(const PatchMatchOptions& options,
                               const PatchMatch::Problem& problem)
    : options_(options),
      problem_(problem),
      ref_width_(0),
      ref_height_(0),
      rotation_in_half_pi_(0) {
  SetBestCudaDevice(std::stoi(options_.gpu_index));
  InitRefImage();
  InitSourceImages();
  InitTransforms();
  InitWorkspaceMemory();
}

PatchMatchCuda::~PatchMatchCuda() {
  for (size_t i = 0; i < 4; ++i) {
    poses_device_[i].reset();
  }
}

void PatchMatchCuda::Run() {
#define CASE_WINDOW_RADIUS(window_radius, window_step)                        \
  case window_radius:                                                         \
    if (options_.pm_algo == "COLMAP"){                                        \
      std::cout << "Using COLMAP Algo." << std::endl;                         \
      RunWithWindowSizeAndStep<2 * window_radius + 1, window_step>();         \
    }                                                                         \
    else if (options_.pm_algo == "ACMM"){                                     \
      std::cout << "Using ACMM Algo." << std::endl;                           \
      ACMMRunWithWindowSizeAndStep<2 * window_radius + 1, window_step>();     \
    }                                                                         \
    else std::cerr << "Error: Algo not supported" << std::endl;               \
    break;

#define CASE_WINDOW_STEP(window_step)                                 \
  case window_step:                                                   \
    switch (options_.window_radius) {                                 \
      CASE_WINDOW_RADIUS(1, window_step)                              \
      CASE_WINDOW_RADIUS(2, window_step)                              \
      CASE_WINDOW_RADIUS(3, window_step)                              \
      CASE_WINDOW_RADIUS(4, window_step)                              \
      CASE_WINDOW_RADIUS(5, window_step)                              \
      CASE_WINDOW_RADIUS(6, window_step)                              \
      CASE_WINDOW_RADIUS(7, window_step)                              \
      CASE_WINDOW_RADIUS(8, window_step)                              \
      CASE_WINDOW_RADIUS(9, window_step)                              \
      CASE_WINDOW_RADIUS(10, window_step)                             \
      CASE_WINDOW_RADIUS(11, window_step)                             \
      CASE_WINDOW_RADIUS(12, window_step)                             \
      CASE_WINDOW_RADIUS(13, window_step)                             \
      CASE_WINDOW_RADIUS(14, window_step)                             \
      CASE_WINDOW_RADIUS(15, window_step)                             \
      CASE_WINDOW_RADIUS(16, window_step)                             \
      CASE_WINDOW_RADIUS(17, window_step)                             \
      CASE_WINDOW_RADIUS(18, window_step)                             \
      CASE_WINDOW_RADIUS(19, window_step)                             \
      CASE_WINDOW_RADIUS(20, window_step)                             \
      default: {                                                      \
        std::cerr << "Error: Window size not supported" << std::endl; \
        break;                                                        \
      }                                                               \
    }                                                                 \
    break;

  switch (options_.window_step) {
    CASE_WINDOW_STEP(1)
    CASE_WINDOW_STEP(2)
    default: {
      std::cerr << "Error: Window step not supported" << std::endl;
      break;
    }
  }

#undef SWITCH_WINDOW_RADIUS
#undef CALL_RUN_FUNC
}

DepthMap PatchMatchCuda::GetDepthMap() const {
  return DepthMap(depth_map_->CopyToMat(), options_.depth_min,
                  options_.depth_max);
}

NormalMap PatchMatchCuda::GetNormalMap() const {
  return NormalMap(normal_map_->CopyToMat());
}

Mat<float> PatchMatchCuda::GetSelProbMap() const {
  return prev_sel_prob_map_->CopyToMat();
}

std::vector<int> PatchMatchCuda::GetConsistentImageIdxs() const {
  const Mat<uint8_t> mask = consistency_mask_->CopyToMat();
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(mask.GetDepth());
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < mask.GetDepth(); ++d) {
        if (mask.Get(r, c, d)) {
          pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
        }
      }
      if (pixel_consistent_image_idxs.size() > 0) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

template <int kWindowSize, int kWindowStep>
void PatchMatchCuda::RunWithWindowSizeAndStep() {
  // Wait for all initializations to finish.
  CUDA_SYNC_AND_CHECK();

  CudaTimer total_timer;
  CudaTimer init_timer;

  ComputeCudaConfig();
  ComputeInitialCost<kWindowSize, kWindowStep>
      <<<sweep_grid_size_, sweep_block_size_>>>(
          *cost_map_, *depth_map_, *normal_map_, *ref_image_->sum_image,
          *ref_image_->squared_sum_image, options_.sigma_spatial,
          options_.sigma_color);
  CUDA_SYNC_AND_CHECK();

  init_timer.Print("Initialization");

  const float total_num_steps = options_.num_iterations * 4;

  SweepOptions sweep_options;
  sweep_options.depth_min = options_.depth_min;
  sweep_options.depth_max = options_.depth_max;
  sweep_options.sigma_spatial = options_.sigma_spatial;
  sweep_options.sigma_color = options_.sigma_color;
  sweep_options.num_samples = options_.num_samples;
  sweep_options.ncc_sigma = options_.ncc_sigma;
  sweep_options.min_triangulation_angle =
      DEG2RAD(options_.min_triangulation_angle);
  sweep_options.incident_angle_sigma = options_.incident_angle_sigma;
  sweep_options.geom_consistency_regularizer =
      options_.geom_consistency_regularizer;
  sweep_options.geom_consistency_max_cost = options_.geom_consistency_max_cost;
  sweep_options.filter_min_ncc = options_.filter_min_ncc;
  sweep_options.filter_min_triangulation_angle =
      DEG2RAD(options_.filter_min_triangulation_angle);
  sweep_options.filter_min_num_consistent = options_.filter_min_num_consistent;
  sweep_options.filter_geom_consistency_max_cost =
      options_.filter_geom_consistency_max_cost;

  for (int iter = 0; iter < options_.num_iterations; ++iter) {
    CudaTimer iter_timer;

    for (int sweep = 0; sweep < 4; ++sweep) {
      CudaTimer sweep_timer;

      // Expenentially reduce amount of perturbation during the optimization.
      sweep_options.perturbation = 1.0f / std::pow(2.0f, iter + sweep / 4.0f);

      // Linearly increase the influence of previous selection probabilities.
      sweep_options.prev_sel_prob_weight =
          static_cast<float>(iter * 4 + sweep) / total_num_steps;

      const bool last_sweep = iter == options_.num_iterations - 1 && sweep == 3;

#define CALL_SWEEP_FUNC                                                  \
  SweepFromTopToBottom<kWindowSize, kWindowStep, kGeomConsistencyTerm,   \
                       kFilterPhotoConsistency, kFilterGeomConsistency>  \
      <<<sweep_grid_size_, sweep_block_size_>>>(                         \
          *global_workspace_, *rand_state_map_, *cost_map_, *depth_map_, \
          *normal_map_, *consistency_mask_, *sel_prob_map_,              \
          *prev_sel_prob_map_, *ref_image_->sum_image,                   \
          *ref_image_->squared_sum_image, sweep_options);

      if (last_sweep) {
        if (options_.filter) {
          consistency_mask_.reset(new GpuMat<uint8_t>(cost_map_->GetWidth(),
                                                      cost_map_->GetHeight(),
                                                      cost_map_->GetDepth()));
          consistency_mask_->FillWithScalar(0);
        }
        if (options_.geom_consistency) {
          const bool kGeomConsistencyTerm = true;
          if (options_.filter) {
            const bool kFilterPhotoConsistency = true;
            const bool kFilterGeomConsistency = true;
            CALL_SWEEP_FUNC
          } else {
            const bool kFilterPhotoConsistency = false;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          }
        } else {
          const bool kGeomConsistencyTerm = false;
          if (options_.filter) {
            const bool kFilterPhotoConsistency = true;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          } else {
            const bool kFilterPhotoConsistency = false;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          }
        }
      } else {
        const bool kFilterPhotoConsistency = false;
        const bool kFilterGeomConsistency = false;
        if (options_.geom_consistency) {
          const bool kGeomConsistencyTerm = true;
          CALL_SWEEP_FUNC
        } else {
          const bool kGeomConsistencyTerm = false;
          CALL_SWEEP_FUNC
        }
      }

#undef CALL_SWEEP_FUNC

      CUDA_SYNC_AND_CHECK();

      Rotate();

      // Rotate selected image map.
      if (last_sweep && options_.filter) {
        std::unique_ptr<GpuMat<uint8_t>> rot_consistency_mask_(
            new GpuMat<uint8_t>(cost_map_->GetWidth(), cost_map_->GetHeight(),
                                cost_map_->GetDepth()));
        consistency_mask_->Rotate(rot_consistency_mask_.get());
        consistency_mask_.swap(rot_consistency_mask_);
      }

      sweep_timer.Print(" Sweep " + std::to_string(sweep + 1));
    }

    iter_timer.Print("Iteration " + std::to_string(iter + 1));
  }

  total_timer.Print("Total");
}

////////////////////////////////////////////////////////////////////////////////////////   ACMM ///////////////////////////////////////////////////////////////////////////////////////////
template <int kWindowSize>
__device__ inline void ACMMReadRefImageIntoSharedMemory(float* local_image,
                                                          const int row,
                                                          const int col,
                                                          const int thread_idx_x,
                                                          const int thread_idx_y) {
  // For the first row, read the entire block into shared memory. For all
  // consecutive rows, it is only necessary to shift the rows in shared memory
  // up by one element and then read in a new row at the bottom of the shared
  // memory. Note that this assumes that the calling loop starts with the first
  // row and then consecutively reads in a new row.  
  const int kWindowRadius = kWindowSize / 2;
  const int base_start_idx = kWindowRadius * (THREADS_PER_BLOCK + 2 * kWindowRadius);
  const int cur_idx = base_start_idx + thread_idx_y * (THREADS_PER_BLOCK + 2 * kWindowRadius) + kWindowRadius + thread_idx_x;
  
  //read the left data exceed the local image
  if(thread_idx_x == 0){
    for(int i = 1; i <= kWindowRadius; i++){
      local_image[cur_idx - i] = tex2D(ref_image_texture, col - i, row);
    }
  }

  //read the right data exceed the local image
  if(thread_idx_x == THREADS_PER_BLOCK - 1){
    for(int i = 1; i <= kWindowRadius; i++){
      local_image[cur_idx + i] = tex2D(ref_image_texture, col + i, row);
    }
  }

  //read the up data exceed the local image
  if(thread_idx_y == 0){
    for(int i = 1; i <= kWindowRadius; i++){
      local_image[cur_idx - i * (THREADS_PER_BLOCK + 2 * kWindowRadius)] = tex2D(ref_image_texture, col, row - i);
    }
  }

  //read the down data exceed the local image
  if(thread_idx_y == THREADS_PER_BLOCK - 1){
    for(int i = 1; i <= kWindowRadius; i++){
      local_image[cur_idx + i * (THREADS_PER_BLOCK + 2 * kWindowRadius)] = tex2D(ref_image_texture, col, row + i);
    }
  }

  //read the left up data exceed the local image
  if(thread_idx_x == 0 && thread_idx_y == 0){
    for(int i = 1; i <= kWindowRadius; i++){
      for(int j = 1; j <= kWindowRadius; j++)
        local_image[cur_idx - i * (THREADS_PER_BLOCK + 2 * kWindowRadius) - j] = tex2D(ref_image_texture, col - j, row - i);
    }
  }

  //read the left down data exceed the local image
  if(thread_idx_x == 0 && thread_idx_y == THREADS_PER_BLOCK - 1){
    for(int i = 1; i <= kWindowRadius; i++){
      for(int j = 1; j <= kWindowRadius; j++)
        local_image[cur_idx + i * (THREADS_PER_BLOCK + 2 * kWindowRadius) - j] = tex2D(ref_image_texture, col - j, row + i);
    }
  }

  //read the right up data exceed the local image
  if(thread_idx_x == THREADS_PER_BLOCK - 1 && thread_idx_y == 0){
    for(int i = 1; i <= kWindowRadius; i++){
      for(int j = 1; j <= kWindowRadius; j++)
        local_image[cur_idx - i * (THREADS_PER_BLOCK + 2 * kWindowRadius) + j] = tex2D(ref_image_texture, col + j, row - i);
    }
  }

  //read the right down data exceed the local image
  if(thread_idx_x == THREADS_PER_BLOCK - 1 && thread_idx_y == THREADS_PER_BLOCK - 1){
    for(int i = 1; i <= kWindowRadius; i++){
      for(int j = 1; j <= kWindowRadius; j++)
        local_image[cur_idx + i * (THREADS_PER_BLOCK + 2 * kWindowRadius) + j] = tex2D(ref_image_texture, col + j, row + i);
    }
  }

  // this pixel
  local_image[cur_idx] = tex2D(ref_image_texture, col, row);

  //sync
  __syncthreads();
}

template <int kWindowSize, int kWindowStep>
__global__ void ACMMComputeInitialCost(GpuMat<float> cost_map,
                                        GpuMat<float> depth_map,
                                        const GpuMat<float> normal_map,
                                        const GpuMat<float> ref_sum_image,
                                        const GpuMat<float> ref_squared_sum_image,
                                        const float sigma_spatial,
                                        const float sigma_color) {
    const int thread_Idx_x = threadIdx.x;
    const int thread_Idx_y = threadIdx.y;

    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
  
    __shared__ float local_ref_image[(THREADS_PER_BLOCK + 2 * (kWindowSize / 2)) * (THREADS_PER_BLOCK + 2 * (kWindowSize / 2))];

    PhotoConsistencyCostComputer<kWindowSize, kWindowStep> pcc_computer(
      sigma_spatial, sigma_color);
    pcc_computer.local_ref_image = local_ref_image;
    pcc_computer.row = row;
    pcc_computer.col = col;

    float normal[3];
    pcc_computer.normal = normal;


    // // Note that this must be executed even for pixels outside the borders,
    // // since pixels are used in the local neighborhood of the current pixel.

    ACMMReadRefImageIntoSharedMemory<kWindowSize>(local_ref_image, row, col, thread_Idx_x, thread_Idx_y);

    if (col < cost_map.GetWidth() && row < cost_map.GetHeight()) {
        pcc_computer.depth = depth_map.Get(row, col);
        normal_map.GetSlice(row, col, normal);
        
        pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
        pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

        for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
          pcc_computer.src_image_idx = image_idx;
          cost_map.Set(row, col, image_idx, pcc_computer.ACMMCompute_shared(thread_Idx_x, thread_Idx_y));
        }
    }
}

template <typename T>
__device__ inline T Get_cu(T* array_ptr_, const size_t row, const size_t col,
                            const size_t slice, const size_t height_, const size_t pitch_) {
  return *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col);
}

template <typename T>
__device__ inline void Set_cu(T* array_ptr_, const size_t row, const size_t col,
                            const size_t slice, const size_t height_, const size_t pitch_, T value) {
   *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col) = value;
}

__device__ inline float AvgCost(float* cost_map, const size_t cost_map_pitch, const size_t cost_map_height, const size_t cost_map_depth, int row, int col){
  float cur_cost = 0.0f;
  //printf("%d\n", cost_map_depth);
  for(size_t image_idx = 0; image_idx < cost_map_depth; image_idx++){
    cur_cost += Get_cu<float>(cost_map, row, col, image_idx, cost_map_height, cost_map_pitch);
    //cur_cost += image_idx;
  }
  cur_cost /= cost_map_depth;
  return cur_cost;
}

__device__ inline void adjustFloatMaxHeap(float * minCost, int * pt_index, int root) {
  while (root < 8) {
    int lch = 2 * root + 1;
    int rch = lch + 1;
    int index = root;

    if (rch < 8 && (minCost[rch] > minCost[index]) ) {
      index = rch;
    }
    if (lch < 8 && (minCost[lch] > minCost[index]) ) {
      index = lch;
    }

    if (index != root) {

      float tmp = minCost[index];
      minCost[index] = minCost[root];
      minCost[root] = tmp;

      int pt_row = pt_index[2 * index];
      int pt_col = pt_index[2 * index + 1];
      pt_index[2 * index] = pt_index[2 * root];
      pt_index[2 * index + 1] = pt_index[2 * root + 1];
      pt_index[2 * root] = pt_row;
      pt_index[2 * root + 1] = pt_col;

      root = index;
    }
    else {
      break;
    }
  }
}

__device__ inline void CheckBoardSampler(float* cost_map,
                                          size_t cost_map_pitch, 
                                          int row, int col,
                                          const size_t rows, const size_t cols, int neighbors,
                                          int upVstep, int downVstep, int leftVstep, int rightVstep,
                                          int upStripStep, int downStripStep, int leftStripStep, int rightStripStep,
                                          float* minCost,
                                          int* pt_index){
  {
    //UP
    //Up V sample
    if (row - 1 >= 0) {
      float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - 1, col);
      if (cost < minCost[0]) {
        minCost[0] = cost;
        pt_index[0] = row - 1;
        pt_index[1] = col;
        adjustFloatMaxHeap(minCost, pt_index, 0);
      }
    }
    for (int i = 1; i <= upVstep; i++) {
      if (row - 1 - i >= 0) {
        if (col - i >= 0) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - 1 - i, col - i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row - 1 - i;
            pt_index[1] = col - i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
        if (col + i < cols) {
           float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - 1 - i, col + i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row - 1 - i;
            pt_index[1] = col + i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
      }
      else {
        break;
      }
    }
    //Up Strip Sample
    for (int i = 0; i < upStripStep; i++) {
      if (row - 3 - 2 * i >= 0) {
        float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - 3 - 2 * i, col);
        if (cost < minCost[0]) {
          minCost[0] = cost;
          pt_index[0] = row - 3 - 2 * i;
          pt_index[1] = col;
          adjustFloatMaxHeap(minCost, pt_index, 0);
        }
      }
      else {
        break;
      }
    }
  }
  {
    //DOWN
    if (row + 1 < rows) {
      float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + 1, col);
      if (cost < minCost[0]) {
        minCost[0] = cost;
        pt_index[0] = row + 1;
        pt_index[1] = col;
        adjustFloatMaxHeap(minCost, pt_index, 0);
      }
    }
    for (int i = 1; i <= downVstep; i++) {
      if (row + 1 + i < rows) {
        if (col - i >= 0) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + 1 + i, col - i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row + 1 + i;
            pt_index[1] = col - i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
        if (col + i < cols) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + 1 + i, col + i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row + 1 + i;
            pt_index[1] = col + i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
      }
      else {
        break;
      }
    }
    //Down Strip sample
    for (int i = 0; i < downStripStep; i++) {
      if (row + 3 + 2 * i < rows) {
        float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + 3 + 2 * i, col);
        if (cost < minCost[0]) {
          minCost[0] = cost;
          pt_index[0] = row + 3 + 2 * i;
          pt_index[1] = col;
          adjustFloatMaxHeap(minCost, pt_index, 0);
        }
      }
      else {
        break;
      }
    }
  }
  {
    //LEFT
    //Left V sample
    if (col - 1 >= 0) {
      float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row, col - 1);
      if (cost < minCost[0] ) {
        minCost[0] = cost;
        pt_index[0] = row;
        pt_index[1] = col - 1;
        adjustFloatMaxHeap(minCost, pt_index, 0);
      }
    }
    for (int i = 1; i <= leftVstep; i++) {
      if (col - 1 - i >= 0) {
        if (row - i >= 0) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - i, col - 1 - i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row - i;
            pt_index[1] = col - 1 - i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
        if (row + i < rows) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + i, col - 1 - i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row + i;
            pt_index[1] = col - 1 - i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
      }
      else {
        break;
      }
    }
    //Left Strip sample
    for (int i = 0; i < leftStripStep; i++) {
      if (col - 3 - 2 * i >= 0) {
        float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row, col - 3 - 2 * i);
        if (cost < minCost[0]) {
          minCost[0] = cost;
          pt_index[0] = row;
          pt_index[1] = col - 3 - 2 * i;
          adjustFloatMaxHeap(minCost, pt_index, 0);
        }
      }
      else {
        break;
      }
    }
  }
  {
    //RIGHT
    //Right V sample
    if (col + 1 < cols) {
      float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row, col + 1);
      if (cost < minCost[0]) {
        minCost[0] = cost;
        pt_index[0] = row;
        pt_index[1] = col + 1;
        adjustFloatMaxHeap(minCost, pt_index, 0);
      }
    }
    for (int i = 1; i <= rightVstep; i++) {
      if (col + 1 + i < cols) {
        if (row - i >= 0) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row - i, col + 1 + i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row - i;
            pt_index[1] = col + 1 + i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
        if (row + i < rows) {
          float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row + i, col + 1 + i);
          if (cost < minCost[0]) {
            minCost[0] = cost;
            pt_index[0] = row + i;
            pt_index[1] = col + 1 + i;
            adjustFloatMaxHeap(minCost, pt_index, 0);
          }
        }
      }
      else {
        break;
      }
    }
    //Right Strip sample
    for (int i = 0; i < rightStripStep; i++) {
      if (col + 3 + 2 * i < cols) {
        float cost = AvgCost(cost_map, cost_map_pitch, rows, neighbors, row, col + 3 + 2 * i);
        if (cost < minCost[0]) {
          minCost[0] = cost;
          pt_index[0] = row;
          pt_index[1] = col + 3 + 2 * i;
          adjustFloatMaxHeap(minCost, pt_index, 0);
        }
      }
      else {
        break;
      }
    }
  }
}

template<int kWindowSize, int kWindowStep, bool kGeomConsistencyTerm = false>
__global__ void ACMMCheckerBoard_cu(GpuMat<float> cost_map,
                                          GpuMat<float> depth_map,
                                          GpuMat<float> normal_map,
                                          GpuMat<float> M_map,
                                          GpuMat<int> last_important_view_map,
                                          GpuMat<float> view_weight_map,
                                          const GpuMat<float> ref_sum_image,
                                          const GpuMat<float> ref_squared_sum_image,
                                          GpuMat<curandState> rand_state_map,
                                          int iter,
                                          float sigma_spatial,
                                          float sigma_color,
                                          float depth_min,
                                          float depth_max
                                          bool isBlack,
                                          float geom_lamda = 0.0f) {

    const int thread_Idx_x = threadIdx.x;
    const int thread_Idx_y = threadIdx.y;

    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    __shared__ float local_ref_image[(THREADS_PER_BLOCK + 2 * (kWindowSize / 2)) * (THREADS_PER_BLOCK + 2 * (kWindowSize / 2))];

    PhotoConsistencyCostComputer<kWindowSize, kWindowStep> pcc_computer(
      sigma_spatial, sigma_color);
    pcc_computer.local_ref_image = local_ref_image;
    pcc_computer.row = row;
    pcc_computer.col = col;

    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    
    ACMMReadRefImageIntoSharedMemory<kWindowSize>(local_ref_image, row, col, thread_Idx_x, thread_Idx_y);
    
  
    if(isBlack && ((threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0) || (threadIdx.x % 2 != 0 && threadIdx.y % 2 != 0)) ||
        !isBlack && ((threadIdx.x % 2 != 0 && threadIdx.y % 2 == 0) || (threadIdx.x % 2 == 0 && threadIdx.y % 2 != 0))){
      // process black pixel
      if (col < cost_map.GetWidth() && row < cost_map.GetHeight()) {
          pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
          pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

          // 8 hypo and correspond uv
          float minCost[8] = {pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost, pcc_computer.kMaxCost};
          int pt_index[16] = {row,col, row,col, row,col, row,col, row,col, row,col, row,col, row,col};
          
          // i. select the 8 hypo with min cost
          CheckBoardSampler(cost_map.GetPtr(), cost_map.GetPitch(), row, col, cost_map.GetHeight(), cost_map.GetWidth(), cost_map.GetDepth(), 
                            V_step_.Get(row, col, 0), V_step_.Get(row, col, 1), V_step_.Get(row, col, 2), V_step_.Get(row, col, 3),
                            S_step_.Get(row, col, 0), S_step_.Get(row, col, 1), S_step_.Get(row, col, 2), S_step_.Get(row, col, 3),
                            minCost, pt_index);

          //update the search area
          



          // 9 hypo: 8 selected and the current
          float normals_0[3];
          float normals_1[3];
          float normals_2[3];
          float normals_3[3];
          float normals_4[3];
          float normals_5[3];
          float normals_6[3];
          float normals_7[3];
          float normals_8[3];
          normal_map.GetSlice(pt_index[2 * 0], pt_index[2 * 0 + 1], normals_0);
          normal_map.GetSlice(pt_index[2 * 1], pt_index[2 * 1 + 1], normals_1);
          normal_map.GetSlice(pt_index[2 * 2], pt_index[2 * 2 + 1], normals_2);
          normal_map.GetSlice(pt_index[2 * 3], pt_index[2 * 3 + 1], normals_3);
          normal_map.GetSlice(pt_index[2 * 4], pt_index[2 * 4 + 1], normals_4);
          normal_map.GetSlice(pt_index[2 * 5], pt_index[2 * 5 + 1], normals_5);
          normal_map.GetSlice(pt_index[2 * 6], pt_index[2 * 6 + 1], normals_6);
          normal_map.GetSlice(pt_index[2 * 7], pt_index[2 * 7 + 1], normals_7);
          normal_map.GetSlice(row, col, normals_8);	
          const float* normals[9] = {normals_0, normals_1, normals_2, normals_3, normals_4, normals_5, normals_6, normals_7, normals_8};
          
          // N: neighbor view
          size_t N = cost_map.GetDepth();

          /*
          * ii. Compute M Matrix
          */
          for (size_t i = 0; i < 9; i++) {
            pcc_computer.normal = normals[i];
            if(i == 8){
              pcc_computer.depth = depth_map.Get(row, col);
            }
            else{
              pcc_computer.depth = depth_map.Get(pt_index[2 * i], pt_index[2 * i + 1]);
            }
          
            for (size_t image_idx = 0; image_idx < N; image_idx++) {
                  pcc_computer.src_image_idx = image_idx;
                  float c = pcc_computer.ACMMCompute_shared(thread_Idx_x, thread_Idx_y);
                  M_map.Set(row, col, N * i + image_idx, c);
            }
          }

          /*
          * iii. Computer viewWeight
          */
          float init_good_threshold = 0.8f;
          float bad_threshold = 1.2f;
          int viewWeight_n1 = 2;
          int viewWeight_n2 = 3;
          float viewWeight_alpha = 90.0f;
          float viewWeight_belta = 0.3f;

          float good_threshold =init_good_threshold * exp(-iter * iter / viewWeight_alpha);
          float maxWeight = 0.0f;
          int lastImportant = -1;

          for (size_t image_idx = 0; image_idx < N; image_idx++) {
            float weight = 0.0f;
            int S_good_size = 0;
            float S_good_score = 0.0f;
            int S_bad_size = 0;

            for (size_t i = 0; i < 9; i++) {
              float mij = M_map.Get(row, col, N * i + image_idx);
              //formula (4)
              if (mij < good_threshold) {
                S_good_score += exp(-mij * mij / (2 * viewWeight_belta * viewWeight_belta));
                S_good_size += 1;
              }
              if (mij > bad_threshold) {
                S_bad_size += 1;
              }
            }
      
            int I = 1;
            if (image_idx == last_important_view_map.Get(row, col)) {
              I = 1;
            }
            else {
              I = 0;
            }
            
            //formula (5)
            if (S_good_size > viewWeight_n1 && S_bad_size < viewWeight_n2) {
              S_good_score = S_good_score / S_good_size;
              weight = (I + 1) * S_good_score;
            }
            else {
              weight = 0.2 * (I);
            }
            // update the lastImportant if this is more important view
            if (weight > maxWeight) {
              lastImportant = image_idx;
              maxWeight = weight;
            }
            view_weight_map.Set(row, col, image_idx, weight);
          }
          last_important_view_map.Set(row, col, lastImportant);

          // select the hypo with min cost
          float minScore = pcc_computer.kMaxCost;
          int minHypo = 8;
          float e_depth = 0.0f;
          for (int i = 0; i < 9; i++) {
            float score = 0.0f;
            float weight_sum = 0.0f;
            
            if(kGeomConsistencyTerm){
              // if use geom, get each hypo depth to reproject
              if(i != 8){
                e_depth = depth_map.Get(pt_index[2 * i], pt_index[2 * i + 1]);
              }
              else{
                e_depth = depth_map.Get(row, col);
              }
            }
            for (size_t image_idx = 0; image_idx < N; image_idx++) {
              float mij = M_map.Get(row, col, N * i + image_idx);
              float eij = 0.0f;
              if(kGeomConsistencyTerm){
                // if use geom, compute reproject error
                eij = ComputeGeomConsistencyCost(row, col, e_depth, image_idx, 2.0f);
              }
              float viewWeight = view_weight_map.Get(row, col, image_idx);
              if (mij < pcc_computer.kMaxCost) {
                score += (mij + geom_lamda * eij) * viewWeight;
                weight_sum += viewWeight;
              }
            }

            if (weight_sum != 0.0f) {
              score = score / weight_sum;
              if (score < 0 || score > pcc_computer.kMaxCost) {
                score = pcc_computer.kMaxCost;
              }
              if (score < minScore) {
                minScore = score;
                minHypo = i;
              }
            }
          }
          
          // update the current state
          const float *best_normal = normals[minHypo];
          float best_depth = depth_map.Get(row, col);
          if(minHypo != 8){
            best_depth = depth_map.Get(pt_index[2 * minHypo], pt_index[2 * minHypo + 1]);
          }
          depth_map.Set(row, col, best_depth);
          normal_map.SetSlice(row, col, best_normal);
          for (size_t image_idx = 0; image_idx < N; image_idx++) {
            cost_map.Set(row, col, image_idx, M_map.Get(row, col, N * minHypo + image_idx));
          }
      }
  }
  
  __syncthreads();
}

template<int kWindowSize, int kWindowStep>
__global__ void RefineMent(GpuMat<float> cost_map,
                      GpuMat<float> depth_map,
                                      GpuMat<float> normal_map,
                                      GpuMat<float> M_map,
                                      GpuMat<int> last_important_view_map,
                                      GpuMat<float> view_weight_map,
                                      const GpuMat<float> ref_sum_image,
                                         const GpuMat<float> ref_squared_sum_image,
                                         GpuMat<curandState> rand_state_map,
                      int iter,
                      float sigma_spatial,
                      float sigma_color,
                      float depth_min,
                      float depth_max) {

    const int thread_Idx_x = threadIdx.x;
    const int thread_Idx_y = threadIdx.y;

    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    //__shared__ float local_ref_image[9 * THREADS_PER_BLOCK * THREADS_PER_BLOCK];
    __shared__ float local_ref_image[(THREADS_PER_BLOCK + 2 * (kWindowSize / 2)) * (THREADS_PER_BLOCK + 2 * (kWindowSize / 2))];

    PhotoConsistencyCostComputer<kWindowSize, kWindowStep> pcc_computer(
      sigma_spatial, sigma_color);
    pcc_computer.local_ref_image = local_ref_image;
    pcc_computer.row = row;
    pcc_computer.col = col;

    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    
    //                                          
    ACMMReadRefImageIntoSharedMemory<kWindowSize>(local_ref_image, row, col,thread_Idx_x, thread_Idx_y);

  if (col < cost_map.GetWidth() && row < cost_map.GetHeight()) {	
    //depth_map.Set(row, col, local_ref_image[(thread_Idx_y + kWindowRadius)* (THREADS_PER_BLOCK + 2 * kWindowRadius) + kWindowRadius + thread_Idx_x]);
      pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
      pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

      float perturbation = 1.0f / pow(2.0f, iter);
      curandState rand_state = rand_state_map.Get(row, col);

      float cur_normal[3];
      normal_map.GetSlice(row, col, cur_normal);

      float rnd_normal[3];
      float prb_normal[3];
      
      GenerateRandomNormal(row, col, &rand_state, rnd_normal);
      PerturbNormal(row, col, perturbation * M_PI, cur_normal, &rand_state, prb_normal);

      float cur_depth = depth_map.Get(row, col);
      float rnd_depth;
      float prb_depth;

      rnd_depth = GenerateRandomDepth(depth_min, depth_max, &rand_state);
      prb_depth = PerturbDepth(perturbation, cur_depth, &rand_state);
          

        //0: prb_norm,rnd_depth
       //1: rnd_norm,rnd_depth
       //2: cur_norm,rnd_depth
       //3: prb_norm,prb_depth
       //4: rnd_norm,prb_depth
       //5: cur_norm,prb_depth
       //6: prb_norm,cur_depth
       //7: rnd_norm,cur_depth
      //8: cur_norm,cur_depth	

       const float hypo_depths[9] = {rnd_depth, rnd_depth, rnd_depth, prb_depth, prb_depth, prb_depth, cur_depth, cur_depth, cur_depth};
      const float* normals[9] = {prb_normal, rnd_normal, cur_normal, prb_normal, rnd_normal, cur_normal, prb_normal, rnd_normal, cur_normal};
    
      size_t N = cost_map.GetDepth();
      /*
      * ii. Compute M Matrix
      */
      // float dis_cost = 0.0f;
      // int index = 0;
      // float M[256];
      // float ViewWeight[32];
      for (size_t i = 0; i < 9; i++) {
        pcc_computer.normal = normals[i];
        pcc_computer.depth = hypo_depths[i];
        
        for (size_t image_idx = 0; image_idx < N; image_idx++) {
              pcc_computer.src_image_idx = image_idx;
              float c = pcc_computer.ACMMCompute_shared(thread_Idx_x, thread_Idx_y);
              M_map.Set(row, col, N * i + image_idx, c);
            }
      }

      // for (int i = 0; i < 9; i++) {
      // 	for (int image_idx = 0; image_idx < N; ++image_idx) {
      // 		dis_cost += M_map.Get(row, col, N * i + image_idx);
      // 	}
      // }
      // dis_cost = dis_cost/(9*N);
      //printf("dis_cost:%f\n", dis_cost);
      //depth_map.Set(row, col, dis_cost);

      /*
      * iii. Computer viewWeight
      */
      float init_good_threshold = 0.8f;
      float bad_threshold = 1.2f;
      int viewWeight_n1 = 2;
      int viewWeight_n2 = 3;
      float viewWeight_alpha = 90.0f;
      float viewWeight_belta = 0.3f;

      float good_threshold =init_good_threshold * exp(-iter * iter / viewWeight_alpha);
      float maxWeight = 0.0f;
      int lastImportant = -1;

      for (size_t image_idx = 0; image_idx < N; image_idx++) {

        float weight = 0.0f;
        int S_good_size = 0;
        float S_good_score = 0.0f;
        int S_bad_size = 0;

        for (size_t i = 0; i < 9; i++) {
          float mij = M_map.Get(row, col, N * i + image_idx);
          //formula (4)
          if (mij < good_threshold) {
            S_good_score += exp(-mij * mij / (2 * viewWeight_belta * viewWeight_belta));
            
            S_good_size += 1;
          }
          if (mij > bad_threshold) {
            S_bad_size += 1;
          }
        }

        int I = 1;
        if (image_idx == last_important_view_map.Get(row, col)) {
          I = 1;
        }
        else {
          I = 0;
        }
        //printf("S_good_size %d\n", S_good_size);
        //printf("S_bad_size %d\n", S_bad_size);
        // //formula (5)
        if (S_good_size > viewWeight_n1 && S_bad_size < viewWeight_n2) {
          S_good_score = S_good_score / S_good_size;
          weight = (I + 1) * S_good_score;
        }
        else {
          weight = 0.2 * (I);
        }
    //printf("mij %f\n", weight);
        if (weight > maxWeight) {
          lastImportant = image_idx;
          maxWeight = weight;
        }
        view_weight_map.Set(row, col, image_idx, weight);
      }
      last_important_view_map.Set(row, col, lastImportant);
      
      // dis_cost = 0;
      // for (int image_idx = 0; image_idx < N; ++image_idx) {
      // 	dis_cost += view_weight_map.Get(row, col, image_idx);
      // }
      
      // dis_cost = dis_cost/(N);
      // printf("dis %f\n", dis_cost);
      // depth_map.Set(row, col, dis_cost);

      float minScore = pcc_computer.kMaxCost;
      int minHypo = 8;

      for (int i = 0; i < 9; i++) {
        float score = 0.0f;
        float weight_sum = 0.0f;
        
        for (size_t image_idx = 0; image_idx < N; image_idx++) {
          float mij = M_map.Get(row, col, N * i + image_idx);
          float viewWeight = view_weight_map.Get(row, col, image_idx);
          if (mij < pcc_computer.kMaxCost) {
            score += mij * viewWeight;
            weight_sum += viewWeight;
          }
        }
        if (weight_sum != 0.0f) {
          score = score / weight_sum;
          if (score < 0 || score > pcc_computer.kMaxCost) {
            score = pcc_computer.kMaxCost;
          }

          if (score < minScore) {
            minScore = score;
            minHypo = i;
          }
        }
      }
      //printf("minHypo %d\n", minHypo);
      const float *best_normal = normals[minHypo];
      float best_depth = hypo_depths[minHypo];
      
      depth_map.Set(row, col, best_depth);
      normal_map.SetSlice(row, col, best_normal);
      for (size_t image_idx = 0; image_idx < N; image_idx++) {
        cost_map.Set(row, col, M_map.Get(row, col, N * minHypo + image_idx));
      }
      rand_state_map.Set(row, col, rand_state);
  }
  __syncthreads();
}

template <int kWindowSize, int kWindowStep>
void PatchMatchCuda::ACMMRunWithWindowSizeAndStep() {
  // Wait for all initializations to finish.
    CUDA_SYNC_AND_CHECK();

    CudaTimer total_timer;
    CudaTimer init_timer;

    ComputeCudaConfig();
    // random init and compute cost
    ACMMComputeInitialCost<kWindowSize, kWindowStep>
      <<<elem_wise_grid_size_, elem_wise_block_size_>>>(
          *cost_map_, *depth_map_, *normal_map_, *ref_image_->sum_image,
          *ref_image_->squared_sum_image, options_.sigma_spatial,
          options_.sigma_color);
    CUDA_SYNC_AND_CHECK();

    init_timer.Print("Initialization");

    bool kGeomConsistencyTerm = false;
    if(options_.){
      kGeomConsistencyTerm = true;
    }
    for(int iter = 0; iter < options_.num_iterations; ++iter) {
      CudaTimer iter_timer;
      CUDA_SYNC_AND_CHECK();
      ACMMCheckerBoard_cu<kWindowSize, kWindowStep, kGeomConsistencyTerm><<<elem_wise_grid_size_, elem_wise_block_size_>>>
      ( *cost_map_, *depth_map_, *normal_map_, *M_map_, *last_important_view_map_, *sel_prob_map_, *ref_image_->sum_image, *ref_image_->squared_sum_image, *rand_state_map_, iter, options_.sigma_spatial,
          options_.sigma_color, options_.depth_min, options_.depth_max, true, options_.geom_consistency_regularizer);
      CUDA_SYNC_AND_CHECK();
      ACMMCheckerBoard_cu<kWindowSize, kWindowStep><<<elem_wise_grid_size_, elem_wise_block_size_>>>
      ( *cost_map_, *depth_map_, *normal_map_, *M_map_, *last_important_view_map_, *sel_prob_map_, *ref_image_->sum_image, *ref_image_->squared_sum_image, *rand_state_map_, iter, options_.sigma_spatial,
          options_.sigma_color, options_.depth_min, options_.depth_max, false);
      CUDA_SYNC_AND_CHECK();
      RefineMent<kWindowSize, kWindowStep><<<elem_wise_grid_size_, elem_wise_block_size_>>>
      ( *cost_map_, *depth_map_, *normal_map_, *M_map_, *last_important_view_map_, *sel_prob_map_, *ref_image_->sum_image, *ref_image_->squared_sum_image, *rand_state_map_, iter, options_.sigma_spatial,
          options_.sigma_color, options_.depth_min, options_.depth_max);
      CUDA_SYNC_AND_CHECK();
      iter_timer.Print("Iteration " + std::to_string(iter + 1));
    }

    total_timer.Print("Total");
}
////////////////////////////////////////////////////////////////////////////////////////   ACMM END ///////////////////////////////////////////////////////////////////////////////////////////


void PatchMatchCuda::ComputeCudaConfig() {
  sweep_block_size_.x = THREADS_PER_BLOCK;
  sweep_block_size_.y = 1;
  sweep_block_size_.z = 1;
  sweep_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  sweep_grid_size_.y = 1;
  sweep_grid_size_.z = 1;

  elem_wise_block_size_.x = THREADS_PER_BLOCK;
  elem_wise_block_size_.y = THREADS_PER_BLOCK;
  elem_wise_block_size_.z = 1;
  elem_wise_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.y =
      (depth_map_->GetHeight() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.z = 1;
}

void PatchMatchCuda::InitRefImage() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  ref_width_ = ref_image.GetWidth();
  ref_height_ = ref_image.GetHeight();

  // Upload to device.
  ref_image_.reset(new GpuMatRefImage(ref_width_, ref_height_));
  const std::vector<uint8_t> ref_image_array =
      ref_image.GetBitmap().ConvertToRowMajorArray();
  ref_image_->Filter(ref_image_array.data(), options_.window_radius,
                     options_.window_step, options_.sigma_spatial,
                     options_.sigma_color);

  ref_image_device_.reset(
      new CudaArrayWrapper<uint8_t>(ref_width_, ref_height_, 1));
  ref_image_device_->CopyFromGpuMat(*ref_image_->image);

  // Create texture.
  ref_image_texture.addressMode[0] = cudaAddressModeBorder;
  ref_image_texture.addressMode[1] = cudaAddressModeBorder;
  ref_image_texture.addressMode[2] = cudaAddressModeBorder;
  ref_image_texture.filterMode = cudaFilterModePoint;
  ref_image_texture.normalized = false;
  CUDA_SAFE_CALL(
      cudaBindTextureToArray(ref_image_texture, ref_image_device_->GetPtr()));
}

void PatchMatchCuda::InitSourceImages() {
  // Determine maximum image size.
  size_t max_width = 0;
  size_t max_height = 0;
  for (const auto image_idx : problem_.src_image_idxs) {
    const Image& image = problem_.images->at(image_idx);
    if (image.GetWidth() > max_width) {
      max_width = image.GetWidth();
    }
    if (image.GetHeight() > max_height) {
      max_height = image.GetHeight();
    }
  }

  // Upload source images to device.
  {
    // Copy source images to contiguous memory block.
    const uint8_t kDefaultValue = 0;
    std::vector<uint8_t> src_images_host_data(
        static_cast<size_t>(max_width * max_height *
                            problem_.src_image_idxs.size()),
        kDefaultValue);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
      const Bitmap& bitmap = image.GetBitmap();
      uint8_t* dest = src_images_host_data.data() + max_width * max_height * i;
      for (size_t r = 0; r < image.GetHeight(); ++r) {
        memcpy(dest, bitmap.GetScanline(r), image.GetWidth() * sizeof(uint8_t));
        dest += max_width;
      }
    }

    // Upload to device.
    src_images_device_.reset(new CudaArrayWrapper<uint8_t>(
        max_width, max_height, problem_.src_image_idxs.size()));
    src_images_device_->CopyToDevice(src_images_host_data.data());

    // Create source images texture.
    src_images_texture.addressMode[0] = cudaAddressModeBorder;
    src_images_texture.addressMode[1] = cudaAddressModeBorder;
    src_images_texture.addressMode[2] = cudaAddressModeBorder;
    src_images_texture.filterMode = cudaFilterModeLinear;
    src_images_texture.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(src_images_texture,
                                          src_images_device_->GetPtr()));
  }

  // Upload source depth maps to device.
  if (options_.geom_consistency) {
    const float kDefaultValue = 0.0f;
    std::vector<float> src_depth_maps_host_data(
        static_cast<size_t>(max_width * max_height *
                            problem_.src_image_idxs.size()),
        kDefaultValue);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      const DepthMap& depth_map =
          problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest =
          src_depth_maps_host_data.data() + max_width * max_height * i;
      for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
        memcpy(dest, depth_map.GetPtr() + r * depth_map.GetWidth(),
               depth_map.GetWidth() * sizeof(float));
        dest += max_width;
      }
    }

    src_depth_maps_device_.reset(new CudaArrayWrapper<float>(
        max_width, max_height, problem_.src_image_idxs.size()));
    src_depth_maps_device_->CopyToDevice(src_depth_maps_host_data.data());

    // Create source depth maps texture.
    src_depth_maps_texture.addressMode[0] = cudaAddressModeBorder;
    src_depth_maps_texture.addressMode[1] = cudaAddressModeBorder;
    src_depth_maps_texture.addressMode[2] = cudaAddressModeBorder;
    // TODO: Check if linear interpolation improves results or not.
    src_depth_maps_texture.filterMode = cudaFilterModePoint;
    src_depth_maps_texture.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(src_depth_maps_texture,
                                          src_depth_maps_device_->GetPtr()));
  }
}

void PatchMatchCuda::InitTransforms() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions (counter-clockwise) of calibration matrix.
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < 4; ++i) {
    ref_K_host_[i][0] = ref_image.GetK()[0];
    ref_K_host_[i][1] = ref_image.GetK()[2];
    ref_K_host_[i][2] = ref_image.GetK()[4];
    ref_K_host_[i][3] = ref_image.GetK()[5];
  }

  // Rotated by 90 degrees.
  std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
  std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
  ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

  // Rotated by 180 degrees.
  ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
  ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

  // Rotated by 270 degrees.
  std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
  std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
  ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

  // Extract 1/fx, -cx/fx, fy, -cy/fy.
  for (size_t i = 0; i < 4; ++i) {
    ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
    ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
    ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
    ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
  }

  // Bind 0 degrees version to constant global memory.
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_K, ref_K_host_[0], sizeof(float) * 4, 0,
                                    cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_inv_K, ref_inv_K_host_[0],
                                    sizeof(float) * 4, 0,
                                    cudaMemcpyHostToDevice));

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions of camera poses.
  //////////////////////////////////////////////////////////////////////////////

  float rotated_R[9];
  memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

  float rotated_T[3];
  memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

  // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
  const float R_z90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};

  for (size_t i = 0; i < 4; ++i) {
    const size_t kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
    std::vector<float> poses_host_data(kNumTformParams *
                                       problem_.src_image_idxs.size());
    int offset = 0;
    for (const auto image_idx : problem_.src_image_idxs) {
      const Image& image = problem_.images->at(image_idx);

      const float K[4] = {image.GetK()[0], image.GetK()[2], image.GetK()[4],
                          image.GetK()[5]};
      memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
      offset += 4;

      float rel_R[9];
      float rel_T[3];
      ComputeRelativePose(rotated_R, rotated_T, image.GetR(), image.GetT(),
                          rel_R, rel_T);
      memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
      offset += 9;
      memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
      offset += 3;

      float C[3];
      ComputeProjectionCenter(rel_R, rel_T, C);
      memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
      offset += 3;

      float P[12];
      ComposeProjectionMatrix(image.GetK(), rel_R, rel_T, P);
      memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
      offset += 12;

      float inv_P[12];
      ComposeInverseProjectionMatrix(image.GetK(), rel_R, rel_T, inv_P);
      memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
      offset += 12;
    }

    poses_device_[i].reset(new CudaArrayWrapper<float>(
        kNumTformParams, problem_.src_image_idxs.size(), 1));
    poses_device_[i]->CopyToDevice(poses_host_data.data());

    RotatePose(R_z90, rotated_R, rotated_T);
  }

  poses_texture.addressMode[0] = cudaAddressModeBorder;
  poses_texture.addressMode[1] = cudaAddressModeBorder;
  poses_texture.addressMode[2] = cudaAddressModeBorder;
  poses_texture.filterMode = cudaFilterModePoint;
  poses_texture.normalized = false;
  CUDA_SAFE_CALL(
      cudaBindTextureToArray(poses_texture, poses_device_[0]->GetPtr()));
}

void PatchMatchCuda::InitWorkspaceMemory() {
  rand_state_map_.reset(new GpuMatPRNG(ref_width_, ref_height_));

  depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
  if (options_.geom_consistency) {
    const DepthMap& init_depth_map =
        problem_.depth_maps->at(problem_.ref_image_idx);
    depth_map_->CopyToDevice(init_depth_map.GetPtr(),
                             init_depth_map.GetWidth() * sizeof(float));
  } else {
    depth_map_->FillWithRandomNumbers(options_.depth_min, options_.depth_max,
                                      *rand_state_map_);
  }

  normal_map_.reset(new GpuMat<float>(ref_width_, ref_height_, 3));

  // Note that it is not necessary to keep the selection probability map in
  // memory for all pixels. Theoretically, it is possible to incorporate
  // the temporary selection probabilities in the global_workspace_.
  // However, it is useful to keep the probabilities for the entire image
  // in memory, so that it can be exported.
  sel_prob_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
                                        problem_.src_image_idxs.size()));
  prev_sel_prob_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
                                             problem_.src_image_idxs.size()));
  prev_sel_prob_map_->FillWithScalar(0.5f);

  cost_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
                                    problem_.src_image_idxs.size()));

  // ACMM
  cost_map_->FillWithScalar(2.0f);

  M_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
                                    9 * problem_.src_image_idxs.size()));

  M_map_->FillWithScalar(2.0f);

  last_important_view_map_.reset(new GpuMat<int>(ref_width_, ref_height_));

  last_important_view_map_->FillWithScalar(-1);

  view_weight_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
                                    problem_.src_image_idxs.size()));

  // up down left right
  V_step_.reset(new GpuMat<int>(ref_width_, ref_height_, 4)); 
  V_step_->FillWithScalar(3);
  S_step_.reset(new GpuMat<int>(ref_width_, ref_height_, 4));
  S_step_->FillWithScalar(11);
  
  // ACMM END

  const int ref_max_dim = std::max(ref_width_, ref_height_);
  global_workspace_.reset(
      new GpuMat<float>(ref_max_dim, problem_.src_image_idxs.size(), 2));

  consistency_mask_.reset(new GpuMat<uint8_t>(0, 0, 0));

  ComputeCudaConfig();

  if (options_.geom_consistency) {
    const NormalMap& init_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    normal_map_->CopyToDevice(init_normal_map.GetPtr(),
                              init_normal_map.GetWidth() * sizeof(float));
  } else {
    InitNormalMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        *normal_map_, *rand_state_map_);
  }
}

void PatchMatchCuda::Rotate() {
  rotation_in_half_pi_ = (rotation_in_half_pi_ + 1) % 4;

  size_t width;
  size_t height;
  if (rotation_in_half_pi_ % 2 == 0) {
    width = ref_width_;
    height = ref_height_;
  } else {
    width = ref_height_;
    height = ref_width_;
  }

  // Rotate random map.
  {
    std::unique_ptr<GpuMatPRNG> rotated_rand_state_map(
        new GpuMatPRNG(width, height));
    rand_state_map_->Rotate(rotated_rand_state_map.get());
    rand_state_map_.swap(rotated_rand_state_map);
  }

  // Rotate depth map.
  {
    std::unique_ptr<GpuMat<float>> rotated_depth_map(
        new GpuMat<float>(width, height));
    depth_map_->Rotate(rotated_depth_map.get());
    depth_map_.swap(rotated_depth_map);
  }

  // Rotate normal map.
  {
    RotateNormalMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        *normal_map_);
    std::unique_ptr<GpuMat<float>> rotated_normal_map(
        new GpuMat<float>(width, height, 3));
    normal_map_->Rotate(rotated_normal_map.get());
    normal_map_.swap(rotated_normal_map);
  }

  // Rotate reference image.
  {
    std::unique_ptr<GpuMatRefImage> rotated_ref_image(
        new GpuMatRefImage(width, height));
    ref_image_->image->Rotate(rotated_ref_image->image.get());
    ref_image_->sum_image->Rotate(rotated_ref_image->sum_image.get());
    ref_image_->squared_sum_image->Rotate(
        rotated_ref_image->squared_sum_image.get());
    ref_image_.swap(rotated_ref_image);
  }

  // Bind rotated reference image to texture.
  ref_image_device_.reset(new CudaArrayWrapper<uint8_t>(width, height, 1));
  ref_image_device_->CopyFromGpuMat(*ref_image_->image);
  CUDA_SAFE_CALL(cudaUnbindTexture(ref_image_texture));
  CUDA_SAFE_CALL(
      cudaBindTextureToArray(ref_image_texture, ref_image_device_->GetPtr()));

  // Rotate selection probability map.
  prev_sel_prob_map_.reset(
      new GpuMat<float>(width, height, problem_.src_image_idxs.size()));
  sel_prob_map_->Rotate(prev_sel_prob_map_.get());
  sel_prob_map_.reset(
      new GpuMat<float>(width, height, problem_.src_image_idxs.size()));

  // Rotate cost map.
  {
    std::unique_ptr<GpuMat<float>> rotated_cost_map(
        new GpuMat<float>(width, height, problem_.src_image_idxs.size()));
    cost_map_->Rotate(rotated_cost_map.get());
    cost_map_.swap(rotated_cost_map);
  }

  // Rotate transformations.
  CUDA_SAFE_CALL(cudaUnbindTexture(poses_texture));
  CUDA_SAFE_CALL(cudaBindTextureToArray(
      poses_texture, poses_device_[rotation_in_half_pi_]->GetPtr()));

  // Rotate calibration.
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_K, ref_K_host_[rotation_in_half_pi_],
                                    sizeof(float) * 4, 0,
                                    cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(ref_inv_K, ref_inv_K_host_[rotation_in_half_pi_],
                         sizeof(float) * 4, 0, cudaMemcpyHostToDevice));

  // Recompute Cuda configuration for rotated reference image.
  ComputeCudaConfig();
}

}  // namespace mvs
}  // namespace colmap
