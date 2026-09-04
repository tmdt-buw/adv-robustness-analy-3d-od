#!/bin/bash
#SBATCH --job-name=compile_mmcv
#SBATCH --partition=gpushort
#SBATCH --account=etechnik_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --time=0-02:00:00
#SBATCH -o %x_logs/%x-%j.out

set -e

#module unload cuda
#module load cuda/12.4


echo "==================================================================="
echo "Compiling MMCV for CLAIX"
echo "==================================================================="

PROJECT_ROOT="/beegfs/krink/Projects/adv-robustness-analy-3d-od/"
cd "$PROJECT_ROOT/external_libs/mmcv"

echo "=== Step 1: Cleaning local build files ==="
rm -rf build/ dist/ *.egg-info
find . -name "*.o" -delete
find . -name "*.so" -delete
echo "✓ Cleaned"

echo "=== Step 2: Injecting CUDA V2 gradient function ==="
CUDA_FILE="mmcv/ops/csrc/pytorch/cuda/voxelization_cuda.cu"
cp "$CUDA_FILE" "${CUDA_FILE}.backup"

cat > "$CUDA_FILE" << 'CUDA_CONTENT'
// Copyright (c) OpenMMLab. All rights reserved.
// Modified to support gradient flow for adversarial attacks
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "voxelization_cuda_kernel.cuh"

// ============================================================================
// NEW 7-tensor signature (for gradient-enabled voxelization)
// ============================================================================
int HardVoxelizeForwardCUDAKernelLauncher_v2(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, at::Tensor &point_to_pointidx,
    at::Tensor &point_to_voxelidx, at::Tensor &coor_to_voxelidx,
    const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim) {

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "hard_voxelize_kernel", ([&] {
    dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(),
        voxel_x, voxel_y, voxel_z, coors_x_min,
        coors_y_min, coors_z_min, coors_x_max,
        coors_y_max, coors_z_max, grid_x, grid_y,
        grid_z, num_points, num_features, NDim);
  }));

  AT_CUDA_CHECK(cudaGetLastError());

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);
  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "point_to_voxelidx", ([&] {
    point_to_voxelidx_kernel<int><<<map_grid, map_block, 0, stream>>>(
        temp_coors.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        point_to_pointidx.contiguous().data_ptr<int>(), max_points,
        max_voxels, num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num = at::zeros({1}, points.options().dtype(at::kInt));

  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_voxel_num", ([&] {
    determin_voxel_num<int><<<1, 1, 0, stream>>>(
        num_points_per_voxel.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        point_to_pointidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxel_num.contiguous().data_ptr<int>(), max_points, max_voxels,
        num_points);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_point_to_voxel", ([&] {
    assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, stream>>>(
        pts_output_size, points.contiguous().data_ptr<float>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxels.contiguous().data_ptr<float>(), max_points, num_features,
        num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_voxel_coors", ([&] {
    assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0, stream>>>(
        coors_output_size, temp_coors.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        coors.contiguous().data_ptr<int>(), num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  return voxel_num_int;
}

// ============================================================================
// OLD 4-tensor signature (backward compatibility)
// ============================================================================
int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim) {

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));
  at::Tensor point_to_pointidx =
      at::zeros({num_points}, points.options().dtype(at::kInt));
  at::Tensor point_to_voxelidx =
      at::zeros({num_points}, points.options().dtype(at::kInt));
  // Must be sized by num_points (not grid dims). determin_voxel_num,
  // assign_point_to_voxel, and assign_voxel_coors all index this tensor by
  // POINT INDEX. The original at::full({grid_z, grid_y, grid_x}, ...)
  // allocates only grid_z*grid_y*grid_x elements (e.g. 1*400*400 = 160000 for
  // the default PointPillars grid), so any input with > 160K points (very
  // common for 10-sweep NuScenes samples) triggers out-of-bounds GPU writes
  // and a "CUDA error: an illegal memory access was encountered" crash inside
  // hard_voxelize_forward. Sized by num_points it is safe.
  at::Tensor coor_to_voxelidx =
      at::full({num_points}, -1, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "hard_voxelize_kernel", ([&] {
    dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(),
        voxel_x, voxel_y, voxel_z, coors_x_min,
        coors_y_min, coors_z_min, coors_x_max,
        coors_y_max, coors_z_max, grid_x, grid_y,
        grid_z, num_points, num_features, NDim);
  }));

  AT_CUDA_CHECK(cudaGetLastError());

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);
  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "point_to_voxelidx", ([&] {
    point_to_voxelidx_kernel<int><<<map_grid, map_block, 0, stream>>>(
        temp_coors.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        point_to_pointidx.contiguous().data_ptr<int>(), max_points,
        max_voxels, num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num = at::zeros({1}, points.options().dtype(at::kInt));

  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_voxel_num", ([&] {
    determin_voxel_num<int><<<1, 1, 0, stream>>>(
        num_points_per_voxel.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        point_to_pointidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxel_num.contiguous().data_ptr<int>(), max_points, max_voxels,
        num_points);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_point_to_voxel", ([&] {
    assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, stream>>>(
        pts_output_size, points.contiguous().data_ptr<float>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxels.contiguous().data_ptr<float>(), max_points, num_features,
        num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_voxel_coors", ([&] {
    assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0, stream>>>(
        coors_output_size, temp_coors.contiguous().data_ptr<int>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        coors.contiguous().data_ptr<int>(), num_points, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  return voxel_num_int;
}

// ============================================================================
// Nondeterministic version
// ============================================================================
int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim) {
  // For now, just use the deterministic version
  return HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel,
      voxel_size, coors_range, max_points, max_voxels, NDim);
}

// ============================================================================
// Dynamic voxelization
// ============================================================================
void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim) {

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", ([&] {
    dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(),
        voxel_x, voxel_y, voxel_z, coors_x_min,
        coors_y_min, coors_z_min, coors_x_max,
        coors_y_max, coors_z_max, grid_x, grid_y,
        grid_z, num_points, num_features, NDim);
  }));

  AT_CUDA_CHECK(cudaGetLastError());
}
CUDA_CONTENT

echo "=== Step 3: Building MMCV ==="
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0" # A100 and H100
export MAX_JOBS=8

pip install -e . --no-build-isolation
pip install "numpy<2.0.0"
python .dev_scripts/check_installation.py

echo "=== Build Complete ==="
