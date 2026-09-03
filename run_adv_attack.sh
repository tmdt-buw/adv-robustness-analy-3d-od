#!/bin/bash
#SBATCH --job-name=adv_attack_3dod
#SBATCH --partition=c23g            # request partition with GPU nodes (c23g for CLAIX A100/H100)
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --gres=gpu:2                # specify desired number of GPUs per node (e.g. gpu:1 or gpu:2)
#SBATCH --cpus-per-task=24          # request CPU cores per process
#SBATCH --mem=120G                  # request memory
#SBATCH --time=24:00:00             # set run time limit (HH:MM:SS)
#SBATCH --output=logs/%x-%j.txt     # stdout & stderr redirected to logs directory
#SBATCH --account=rwth2049          # account / project id

# ==============================================================================
# Helper Function: Format Seconds to HH:MM:SS
# ==============================================================================
format_time() {
    local T=$1
    local H=$((T/60/60))
    local M=$((T/60%60))
    local S=$((T%60))
    printf "%02d:%02d:%02d" $H $M $S
}

TOTAL_START=$(date +%s)
echo "======================================================================"
echo "Job Started: $(date)"
echo "Node Allocated: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

# Ensure logs directory exists
mkdir -p logs

# 1. Define Paths
SRC_DIR="/hpcwork/rwth2049/clean_data/nuscenes/zipped"
DEST_DIR="$TMPDIR/nuscenes"

mkdir -p "$DEST_DIR"

# ==============================================================================
# PHASE 1: Data Transfer
# ==============================================================================
echo -e "\n[PHASE 1] Transferring .tar.zst archives to local NVMe ($TMPDIR)..."
TRANSFER_START=$(date +%s)

cp "$SRC_DIR/binaries_v1.0.tar.zst" "$TMPDIR/" &
cp "$SRC_DIR/samples_LIDAR_TOP.tar.zst" "$TMPDIR/" &
cp "$SRC_DIR/sweeps_LIDAR_TOP.tar.zst" "$TMPDIR/" &
wait

TRANSFER_END=$(date +%s)
TRANSFER_DUR=$((TRANSFER_END - TRANSFER_START))
echo "Transfer complete in $(format_time $TRANSFER_DUR)."

# ==============================================================================
# PHASE 2: Data Extraction & Verification
# ==============================================================================
echo -e "\n[PHASE 2] Decompressing archives locally in parallel..."
EXTRACT_START=$(date +%s)

# Decompress using the pipe approach, stripping the baked-in parent folder
zstd -d -c "$TMPDIR/binaries_v1.0.tar.zst" | tar -xf - --strip-components=1 -C "$DEST_DIR/" &
zstd -d -c "$TMPDIR/samples_LIDAR_TOP.tar.zst" | tar -xf - --strip-components=1 -C "$DEST_DIR/" &
zstd -d -c "$TMPDIR/sweeps_LIDAR_TOP.tar.zst" | tar -xf - --strip-components=1 -C "$DEST_DIR/" &
cp -r "$SRC_DIR/maps" "$DEST_DIR/" &
wait

if [ -d "$DEST_DIR/binaries" ]; then
    mv "$DEST_DIR/binaries/"* "$DEST_DIR/"
    rmdir "$DEST_DIR/binaries"
fi

# Structural safety check
if [ ! -d "$DEST_DIR/samples" ] || [ ! -d "$DEST_DIR/sweeps" ] || [ ! -d "$DEST_DIR/v1.0-trainval" ]; then
    echo "ERROR: Folder structure mismatch! Core dataset folders are missing."
    ls -la "$DEST_DIR"
    exit 1
fi

# ------------------------------------------------------------------------------
# DIRECTORY TREE LOGGING
# ------------------------------------------------------------------------------
echo -e "\n======================================================================"
echo "Extracted nuScenes Directory Structure (Depth = 2, File Limit = 50):"
echo "======================================================================"
tree -L 2 --filelimit 50 "$DEST_DIR"
echo "======================================================================"

# Clean up NVMe tarballs to free space
rm "$TMPDIR/binaries_v1.0.tar.zst" "$TMPDIR/samples_LIDAR_TOP.tar.zst" "$TMPDIR/sweeps_LIDAR_TOP.tar.zst"

EXTRACT_END=$(date +%s)
EXTRACT_DUR=$((EXTRACT_END - EXTRACT_START))
echo "Extraction and verification complete in $(format_time $EXTRACT_DUR)."

# ==============================================================================
# PHASE 3: Environment Setup & Execution
# ==============================================================================
echo -e "\n[PHASE 3] Setting Up Environment and Launching Attack Pipeline..."
ATTACK_START=$(date +%s)

# Dataset location env variable (picked up by MMDetection3D / MMEngine configs)
export NUSCENES_DATA_ROOT="$DEST_DIR/"

# Hardware and library optimizations (H100 / A100)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1   # TF32 for matmul
export TORCH_CUDNN_V8_API_ENABLED=1         # TF32 for cuDNN convolutions
export NCCL_ASYNC_ERROR_HANDLING=1          # Surface NCCL errors faster
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Prevent CPU thread over-subscription in DataLoader workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Define project directories
PROJ_DIR="/home/fzn38120/Projects/adv-robustness-analy-3d-od"
SAVE_DIR="/home/fzn38120/Projects/adv_data_aug/ECCV2026/visualizations"

mkdir -p "$SAVE_DIR"

module load CUDA
echo "######################## Initializing mamba environment #########################"
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
echo "######################## Activating mamba environment 'mmdet3d' #########################"
mamba activate mmdet3d

# Fix CXXABI / SQLite cluster dynamic loader compatibility
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ------------------------------------------------------------------------------
# Attack Pipeline Configuration
# ------------------------------------------------------------------------------
# Preset models: centerpoint, pillarnest, pointpillars, focalformer3d, custom
MODEL_PRESET="pillarnest"

# Attacks: iou_detachment, iou_attachment, iou_perturbation, fgsm, pgd, lidattack
ATTACK="iou_detachment"

# Dataset mode: set REDUCED="--reduced" for reduced split, or REDUCED="" for full split
REDUCED="--reduced"

# Optional sample count limit (leave empty to iterate all samples)
NUM_SAMPLES="1" # e.g. "--num-samples 100"

# Optional extra arguments (e.g. "--cuda-memory-monitor", "--debug")
EXTRA_ARGS=""

# Multi-GPU check
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s) allocated."

cd "$PROJ_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    RANDOM_PORT=$((10000 + RANDOM % 50000))
    echo "Launching Distributed Attack on $NUM_GPUS GPUs (Port: $RANDOM_PORT)..."
    python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=$RANDOM_PORT \
        "$PROJ_DIR/adversarial_attack_pipeline.py" \
        --preset-model "$MODEL_PRESET" \
        --attack "$ATTACK" \
        $REDUCED \
        $NUM_SAMPLES \
        --save-dir "$SAVE_DIR" \
        --launcher pytorch \
        $EXTRA_ARGS
else
    echo "Launching Single-GPU Attack..."
    python "$PROJ_DIR/adversarial_attack_pipeline.py" \
        --preset-model "$MODEL_PRESET" \
        --attack "$ATTACK" \
        $REDUCED \
        $NUM_SAMPLES \
        --save-dir "$SAVE_DIR" \
        $EXTRA_ARGS
fi

ATTACK_END=$(date +%s)
ATTACK_DUR=$((ATTACK_END - ATTACK_START))
echo "Attack phase concluded in $(format_time $ATTACK_DUR)."

# ==============================================================================
# FINAL SUMMARY PROFILE
# ==============================================================================
TOTAL_END=$(date +%s)
TOTAL_DUR=$((TOTAL_END - TOTAL_START))

echo -e "\n======================================================================"
echo "                   PIPELINE EXECUTION SUMMARY"
echo "======================================================================"
echo " 1. Data Transfer (Network -> NVMe) : $(format_time $TRANSFER_DUR)"
echo " 2. Decompression & Data Staging    : $(format_time $EXTRACT_DUR)"
echo " 3. Adversarial Attack Execution    : $(format_time $ATTACK_DUR)"
echo "----------------------------------------------------------------------"
echo " TOTAL WALL-CLOCK TIME              : $(format_time $TOTAL_DUR)"
echo "======================================================================"
echo "Job finished successfully on $(date)"
