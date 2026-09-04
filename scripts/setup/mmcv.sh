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

#conda init
#conda activate mmdet3d

module unload CUDA
module load CUDA/12.4.0

export MMCV_WITH_OPS=1
export FORCE_CUDA=1

pip install -e . --no-build-isolation
pip install "numpy<2.0.0"
python .dev_scripts/check_installation.py