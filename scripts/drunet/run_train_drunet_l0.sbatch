#!/bin/bash
#SBATCH --job-name=drunet_l0
#SBATCH --partition=GPU48Go
#SBATCH --nodelist=cn60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home4/adepaepe/dev/ct-ml-pnp
#SBATCH --output=logs/slurm/drunet_l0_%j.out      # stdout log
#SBATCH --error=logs/slurm/drunet_l0_%j.err       # stderr log

# Optional: load modules if needed
# module load apptainer
# module load cuda

echo "Running on node $(hostname)"
echo "Starting job $SLURM_JOB_NAME at $(date)"

# Force specific GPU if needed
# export CUDA_VISIBLE_DEVICES=1

# Run the training script
srun apptainer exec --nv /home4/adepaepe/containers/apptainer_deepinv.sif \
    python3 train_drunet_multilevel.py trainings=drunet_l0

echo "Job finished at $(date)"
