#!/bin/bash
#SBATCH --job-name=compile_chamfer
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

echo "Building ChamferDist in $(pwd)"

python setup.py install
echo "=== ChamferDist Build Complete ==="
