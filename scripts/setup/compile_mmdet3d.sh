#!/bin/bash
#SBATCH --job-name=compile_mmdet3d
#SBATCH --partition=gpushort
#SBATCH --account=etechnik_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --time=0-02:00:00
#SBATCH -o %x_logs/%x-%j.out

module unload cuda
module load cuda/12.1

echo "Building MMDetection3D in $(pwd)"


# Use --no-deps --no-build-isolation to compile successfully without internet on the compute node
pip install -e . -v --no-build-isolation

echo "=== MMDetection3D Build Complete ==="
