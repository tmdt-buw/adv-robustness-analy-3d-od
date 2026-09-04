#!/bin/sh
#SBATCH --job-name=visualize_attack
#SBATCH --partition=gpushort
#SBATCH --account=etechnik_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --time=0-06:00:00
#SBATCH -o %x_logs/%x-%j.out


# Set a specific directory with a job-unique name:
workdir="/tmp/${USER:?}_${SLURM_JOB_ID:?}"
submitdir="${SLURM_SUBMIT_DIR:?}"
traindatadirBeeGFS="/beegfs/${USER:?}/data"

mkdir -p "${workdir}"
echo "${workdir}"

function clean_up {
    # Leave ${workdir}
    cd "${submitdir}" || exit
    # Use :? to only remove if the variable is defined. Otherwise exit
    rm -rf "${workdir:?}"
    echo "clean up done at ${workdir}"

    # Remove all directories or files from the directory /tmp that are owned by the current user if there does not exists another run on the same node
    if [ ! -d /tmp/"${USER:?}_" ]; then
        # Iterate through all files and directories in /tmp
        for item in /tmp/*; do
            if [ "$(stat -c %U "$item")" == "${USER:?}" ]; then
                echo "Removing $item"
                rm -rf "$item"
            fi
        done
    else
        echo "Directory /tmp/${USER:?}_ exists, skipping removal of user directories"
    fi
    exit
}

# Always call "clean_up" when script ends
# This even executes on job failure/cancellation
trap 'clean_up' EXIT

# Start real work in workdir
date
echo "copying data.."

echo "load env"
module load 2023a
source /beegfs/krink/miniconda3/bin/activate mmdet3d
date
echo "starting run.."

cd /beegfs/krink/Projects/adv-robustness-analy-3d-od

python data_processing/visualize.py \
    --db-path /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/adversarial_attack.db \
    --output /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations \
    --samples 0 \
    --adv \
    --no-legend
