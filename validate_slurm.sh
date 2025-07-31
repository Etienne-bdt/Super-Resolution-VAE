#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=SR_VAE_Validation
#SBATCH -o ./slurm_logs/validation.%j.out # STDOUT
#SBATCH -e ./slurm_logs/validation.%j.err # STDERR
#SBATCH --partition=gpu02
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

# Load required modules
module load python/3.8

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export SCRATCH="/scratch/disc/e.bardet/"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Check if model checkpoint is provided
if [ -z "$MODEL_CKPT" ]; then
    echo "Error: MODEL_CKPT environment variable not set"
    echo "Usage: MODEL_CKPT=/path/to/checkpoint.pth sbatch validate_slurm.sh"
    exit 1
fi

# Set default values if not provided
MODEL_TYPE=${MODEL_TYPE:-"MVAE"}
DATASET=${DATASET:-"s2v"}
PATCH_SIZE=${PATCH_SIZE:-64}
COMPRESSION_RATIO=${COMPRESSION_RATIO:-1.5}
GAMMA_TYPE=${GAMMA_TYPE:-"scalar"}
L=${L:-1}
NUM_SAMPLES=${NUM_SAMPLES:-100}
BATCH_SIZE=${BATCH_SIZE:-1}

echo "Starting validation..."

# Run validation
python validate.py \
    --model_ckpt "$MODEL_CKPT" \
    --model_type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --patch_size "$PATCH_SIZE" \
    --compression_ratio "$COMPRESSION_RATIO" \
    --gamma_type "$GAMMA_TYPE" \
    --L "$L" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "validation_results"

echo "Validation completed!"