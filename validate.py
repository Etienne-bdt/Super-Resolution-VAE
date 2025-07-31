#!/usr/bin/env python3
"""
Validation script for Super-Resolution VAE models.

This script loads a trained model from a checkpoint and evaluates it on the full validation set,
computing SSIM metrics and identifying best and worst case samples.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import metrics as skmetrics

import models
from dataset import init_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate a Super-Resolution VAE model.")
    
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint to load."
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="MVAE",
        choices=["MVAE", "VAE", "Cond_VAE"],
        help="Model type to instantiate."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="s2v",
        help="Dataset to use for validation."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for validation (recommended: 1 for proper SSIM computation)."
    )
    
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Patch size for the model."
    )
    
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=1.5,
        help="Compression ratio used during training."
    )
    
    parser.add_argument(
        "--gamma_type",
        type=str,
        default="scalar",
        choices=["scalar", "vector"],
        help="Type of gamma used in the model."
    )
    
    parser.add_argument(
        "--L",
        type=int,
        default=1,
        help="Number of latent sampling in the model."
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate for uncertainty estimation."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results."
    )
    
    return parser.parse_args()


def load_model(args) -> torch.nn.Module:
    """Load and instantiate the model from checkpoint."""
    print(f"Loading model type: {args.model_type}")
    print(f"Model checkpoint: {args.model_ckpt}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.model_ckpt):
        raise FileNotFoundError(f"Model checkpoint {args.model_ckpt} not found.")
    
    # Generate unique identifier for this validation run
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"validation_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Instantiate model based on type
    if args.model_type == "VAE":
        model = models.VAE(
            args.compression_ratio,
            args.patch_size // 2,
            callbacks=[],
            slurm_job_id=slurm_job_id,
        )
    elif args.model_type == "MVAE":
        model = models.Multimodal_VAE(
            args.compression_ratio,
            args.patch_size,
            callbacks=[],
            slurm_job_id=slurm_job_id,
            L=args.L,
            gamma_type=args.gamma_type,
        )
    elif args.model_type == "Cond_VAE":
        model = models.Cond_VAE(
            args.compression_ratio,
            args.patch_size,
            callbacks=[],
            slurm_job_id=slurm_job_id,
            L=args.L,
            gamma_type=args.gamma_type,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load checkpoint
    print("Loading model weights from checkpoint...")
    checkpoint = torch.load(args.model_ckpt, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and any(key.startswith('encoder') or key.startswith('decoder') for key in checkpoint.keys()):
        model.load_state_dict(checkpoint)
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully.")
    return model


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between prediction and target."""
    # Convert to numpy and move to RGB format for SSIM computation
    pred_np = pred[[2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
    target_np = target[[2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1] range
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    ssim_value = skmetrics.structural_similarity(
        pred_np, target_np,
        data_range=1.0,
        multichannel=True,
        channel_axis=2
    )
    return ssim_value


def validate_model(model: torch.nn.Module, val_loader, device: torch.device, args) -> Dict:
    """Run validation on the full validation set."""
    model.eval()
    
    results = {
        'ssim_scores': [],
        'sample_indices': [],
        'best_case': {'ssim': -1, 'idx': -1, 'pred': None, 'target': None, 'input': None},
        'worst_case': {'ssim': 2, 'idx': -1, 'pred': None, 'target': None, 'input': None},
        'mean_ssim': 0.0,
        'std_ssim': 0.0,
        'all_samples': []
    }
    
    print(f"Running validation on {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
            # Unpack batch - format depends on dataset
            if len(batch) == 2:
                input_img, target_img = batch
                input_img = input_img.to(device)
                target_img = target_img.to(device)
            else:
                # Handle single image case or other formats
                input_img = batch[0].to(device)
                target_img = batch[1].to(device) if len(batch) > 1 else input_img
            
            # Process each sample in the batch
            batch_size = input_img.size(0)
            for sample_idx in range(batch_size):
                sample_input = input_img[sample_idx:sample_idx+1]
                sample_target = target_img[sample_idx:sample_idx+1]
                
                # Generate samples from the model
                try:
                    # Use the model's sample method if available
                    if hasattr(model, 'sample'):
                        samples = model.sample(sample_input, samples=args.num_samples)
                        pred = samples.mean(dim=0)  # Use mean of samples as prediction
                    else:
                        # Fallback to forward pass
                        pred = model(sample_input)
                        if isinstance(pred, tuple):
                            pred = pred[0]  # Take reconstruction if tuple returned
                except Exception as e:
                    print(f"Error processing sample {batch_idx}-{sample_idx}: {e}")
                    continue
                
                # Compute SSIM
                try:
                    ssim_score = compute_ssim(pred[0], sample_target[0])
                    
                    # Track results
                    global_idx = batch_idx * batch_size + sample_idx
                    results['ssim_scores'].append(ssim_score)
                    results['sample_indices'].append(global_idx)
                    
                    # Update best case
                    if ssim_score > results['best_case']['ssim']:
                        results['best_case'].update({
                            'ssim': ssim_score,
                            'idx': global_idx,
                            'pred': pred[0].cpu().clone(),
                            'target': sample_target[0].cpu().clone(),
                            'input': sample_input[0].cpu().clone()
                        })
                    
                    # Update worst case  
                    if ssim_score < results['worst_case']['ssim']:
                        results['worst_case'].update({
                            'ssim': ssim_score,
                            'idx': global_idx,
                            'pred': pred[0].cpu().clone(),
                            'target': sample_target[0].cpu().clone(),
                            'input': sample_input[0].cpu().clone()
                        })
                        
                except Exception as e:
                    print(f"Error computing SSIM for sample {batch_idx}-{sample_idx}: {e}")
                    continue
    
    # Compute statistics
    if results['ssim_scores']:
        results['mean_ssim'] = np.mean(results['ssim_scores'])
        results['std_ssim'] = np.std(results['ssim_scores'])
    
    return results


def save_results(results: Dict, output_dir: str, args):
    """Save validation results and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    results_file = os.path.join(output_dir, "validation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Validation Results\n")
        f.write(f"==================\n\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Model Checkpoint: {args.model_ckpt}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Compression Ratio: {args.compression_ratio}\n")
        f.write(f"Number of Samples: {len(results['ssim_scores'])}\n\n")
        
        f.write(f"SSIM Statistics:\n")
        f.write(f"Mean SSIM: {results['mean_ssim']:.6f}\n")
        f.write(f"Std SSIM: {results['std_ssim']:.6f}\n")
        f.write(f"Min SSIM: {min(results['ssim_scores']):.6f}\n")
        f.write(f"Max SSIM: {max(results['ssim_scores']):.6f}\n\n")
        
        f.write(f"Best Case (Sample {results['best_case']['idx']}):\n")
        f.write(f"  SSIM: {results['best_case']['ssim']:.6f}\n\n")
        
        f.write(f"Worst Case (Sample {results['worst_case']['idx']}):\n")
        f.write(f"  SSIM: {results['worst_case']['ssim']:.6f}\n\n")
    
    # Plot SSIM distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['ssim_scores'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(results['mean_ssim'], color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {results["mean_ssim"]:.4f}')
    plt.axvline(results['best_case']['ssim'], color='green', linestyle='--', linewidth=2,
                label=f'Best: {results["best_case"]["ssim"]:.4f}')
    plt.axvline(results['worst_case']['ssim'], color='orange', linestyle='--', linewidth=2,
                label=f'Worst: {results["worst_case"]["ssim"]:.4f}')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('SSIM Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "ssim_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save best and worst case visualizations
    def save_comparison(case_data, case_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input (upsampled for comparison)
        input_upsampled = torch.nn.functional.interpolate(
            case_data['input'].unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False
        )[0]
        axes[0].imshow(input_upsampled[[2, 1, 0], :, :].clamp(0, 1).permute(1, 2, 0).numpy())
        axes[0].set_title('Input (Bicubic Upsampled)')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(case_data['pred'][[2, 1, 0], :, :].clamp(0, 1).permute(1, 2, 0).numpy())
        axes[1].set_title(f'Prediction\nSSIM: {case_data["ssim"]:.4f}')
        axes[1].axis('off')
        
        # Ground Truth
        axes[2].imshow(case_data['target'][[2, 1, 0], :, :].clamp(0, 1).permute(1, 2, 0).numpy())
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.suptitle(f'{case_name} Case (Sample {case_data["idx"]})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{case_name.lower()}_case.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    if results['best_case']['pred'] is not None:
        save_comparison(results['best_case'], 'Best')
    
    if results['worst_case']['pred'] is not None:
        save_comparison(results['worst_case'], 'Worst')
    
    print(f"Results saved to {output_dir}")
    print(f"Mean SSIM: {results['mean_ssim']:.6f} ± {results['std_ssim']:.6f}")
    print(f"Best case SSIM: {results['best_case']['ssim']:.6f} (sample {results['best_case']['idx']})")
    print(f"Worst case SSIM: {results['worst_case']['ssim']:.6f} (sample {results['worst_case']['idx']})")


def main():
    """Main validation function."""
    args = parse_args()
    
    print("==========================")
    print("Super-Resolution VAE Validation")
    print("==========================")
    print(f"Model Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Patch Size: {args.patch_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Samples: {args.num_samples}")
    print("--------------------------")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args)
    model.to(device)
    
    # Initialize data loader (only validation set)
    _, val_loader = init_dataloader(args.dataset, args.batch_size, args.patch_size)
    print(f"Validation set size: {len(val_loader)} batches")
    
    # Run validation
    results = validate_model(model, val_loader, device, args)
    
    # Save results
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_validation_{time.strftime('%Y%m%d_%H%M%S')}")
    save_results(results, output_dir, args)


if __name__ == "__main__":
    main()