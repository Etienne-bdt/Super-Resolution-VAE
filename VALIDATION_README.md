# Validation Scripts Usage

This directory contains validation scripts for the Super-Resolution VAE models.

## Files

- `validate.py`: Main validation script that loads a model and evaluates it on the full validation set
- `validate_slurm.sh`: SLURM batch script for running validation on a cluster

## Usage

### Local Validation

```bash
python validate.py \
    --model_ckpt /path/to/your/checkpoint.pth \
    --model_type MVAE \
    --dataset s2v \
    --patch_size 64 \
    --compression_ratio 1.5 \
    --num_samples 100 \
    --output_dir validation_results
```

### SLURM Validation

```bash
# Set environment variables and submit job
MODEL_CKPT=/path/to/your/checkpoint.pth \
MODEL_TYPE=MVAE \
DATASET=s2v \
PATCH_SIZE=64 \
COMPRESSION_RATIO=1.5 \
NUM_SAMPLES=100 \
sbatch validate_slurm.sh
```

## Parameters

- `--model_ckpt`: **Required** Path to the model checkpoint file
- `--model_type`: Model type (VAE, MVAE, Cond_VAE) [default: MVAE]
- `--dataset`: Dataset to use (s2v, Sen2Venus, floods) [default: s2v]
- `--patch_size`: Patch size used during training [default: 64] 
- `--compression_ratio`: Compression ratio used during training [default: 1.5]
- `--gamma_type`: Gamma type (scalar, vector) [default: scalar]
- `--L`: Number of latent samples [default: 1]
- `--num_samples`: Number of samples for uncertainty estimation [default: 100]
- `--batch_size`: Batch size for validation [default: 1]
- `--output_dir`: Directory to save results [default: validation_results]

## Output

The validation script will create a timestamped directory with:

1. `validation_results.txt`: Summary statistics and best/worst case information
2. `ssim_distribution.png`: Histogram of SSIM scores across all validation samples
3. `best_case.png`: Visualization of the sample with highest SSIM
4. `worst_case.png`: Visualization of the sample with lowest SSIM

### Example Output Statistics

```
SSIM Statistics:
Mean SSIM: 0.872543
Std SSIM: 0.045621
Min SSIM: 0.654321
Max SSIM: 0.945123

Best Case (Sample 1234):
  SSIM: 0.945123

Worst Case (Sample 567):
  SSIM: 0.654321
```

## Requirements

The script requires the same environment as the training script, including:
- PyTorch
- scikit-image (for SSIM computation)
- matplotlib (for visualizations)
- numpy
- tqdm

Make sure to activate your virtual environment before running the scripts.