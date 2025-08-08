import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from dataset import init_dataloader
from models import Cond_VAE
from utils import save_img, save_img_histogram


def main():
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local_run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = init_dataloader("s2v", batch_size=1, patch_size=256)
    model = Cond_VAE(
        cr=1.5, 
        patch_size=256, 
        gamma_type="scalar", 
        slurm_job_id=slurm_job_id,
        distribution_type="laplacian"  # Default to laplacian for backward compatibility
    )

    ckpt = torch.load("ckpt/3871425.pth", map_location="cpu")
    print(ckpt.keys())
    model.load_state_dict(ckpt)
    model.eval()

    model.to(device)

    bicubic = np.zeros(len(val_loader), dtype=np.float32)
    model_store = np.zeros(len(val_loader), dtype=np.float32)
    for i, (y, x) in tqdm(enumerate(val_loader), total=len(val_loader)):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out, *_ = model.sample(y, 1, gamma_added=False)

        bicubic_upsampled = F.interpolate(
            y,
            scale_factor=2,
            mode="bicubic",
        )

        bicubic[i] = ssim(
            x[0, :, :, :].cpu().numpy(),
            bicubic_upsampled[0, :, :, :].cpu().numpy(),
            data_range=1.0,
            multichannel=True,
            channel_axis=0,
        )
        model_store[i] = ssim(
            out[:, :, :].cpu().numpy(),
            x[0, :, :, :].cpu().numpy(),
            data_range=1.0,
            multichannel=True,
            channel_axis=0,
        )

    print(f"Bicubic SSIM: {bicubic.mean().item()}")
    print(f"Model SSIM: {model_store.mean().item()}")
    print(f"Improvement: {model_store.mean().item() - bicubic.mean().item()}")

    # Sort indices by SSIM score
    sorted_indices = np.argsort(model_store)  # ascending order (worst to best)

    # Select which rank to save (0=worst, 1=2nd worst, etc.)
    worst_rank = 5  # Change this to select different worst ranks
    best_rank = 5  # Change this to select different best ranks

    idx_min = int(sorted_indices[int(worst_rank)])
    idx_max = int(sorted_indices[-int(best_rank + 1)])  # negative indexing for best

    print(f"Worst SSIM (rank {worst_rank + 1}): {model_store[idx_min].item()}")
    print(f"Bicubic at that index: {bicubic[idx_min].item()}")

    print(f"Best SSIM (rank {best_rank + 1}): {model_store[idx_max].item()}")
    print(f"Bicubic at that index: {bicubic[idx_max].item()}")

    # Print top 5 worst and best for reference
    print("\nTop 5 worst SSIM scores:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  Rank {i + 1}: SSIM={model_store[idx]:.4f}, Index={idx}")

    print("\nTop 5 best SSIM scores:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[-(i + 1)]
        print(f"  Rank {i + 1}: SSIM={model_store[idx]:.4f}, Index={idx}")

    # Save the worst and best images
    worst_y, worst_x = val_loader.dataset[idx_min]
    model_worse = model.sample(worst_y.unsqueeze(0).to(device), 1, gamma_added=True, recurrent=True)[
        0
    ].detach()
    worst_bicubic = F.interpolate(
        worst_y.unsqueeze(0).to(device),
        scale_factor=2,
        mode="bicubic",
    )[0].detach()

    best_y, best_x = val_loader.dataset[idx_max]
    model_best = model.sample(best_y.unsqueeze(0).to(device), 1, gamma_added=True, recurrent=True)[
        0
    ].detach()
    best_bicubic = F.interpolate(
        best_y.unsqueeze(0).to(device),
        scale_factor=2,
        mode="bicubic",
    )[0].detach()

    os.makedirs(f"results/{slurm_job_id}", exist_ok=True)

    # Save regular color versions
    save_img(worst_x, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_x.png")
    save_img(worst_y, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_y.png")
    save_img(
        worst_bicubic, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_bicubic.png"
    )
    save_img(
        model_worse, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_model.png"
    )
    save_img(best_x, f"results/{slurm_job_id}/best_rank{best_rank + 1}_x.png")
    save_img(best_y, f"results/{slurm_job_id}/best_rank{best_rank + 1}_y.png")
    save_img(
        best_bicubic, f"results/{slurm_job_id}/best_rank{best_rank + 1}_bicubic.png"
    )
    save_img(model_best, f"results/{slurm_job_id}/best_rank{best_rank + 1}_model.png")

    # Save false color versions
    save_img(
        worst_x,
        f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_x_false_color.png",
        false_color=True,
    )
    save_img(
        worst_y,
        f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_y_false_color.png",
        false_color=True,
    )
    save_img(
        worst_bicubic,
        f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_bicubic_false_color.png",
        false_color=True,
    )
    save_img(
        model_worse,
        f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_model_false_color.png",
        false_color=True,
    )
    save_img(
        best_x,
        f"results/{slurm_job_id}/best_rank{best_rank + 1}_x_false_color.png",
        false_color=True,
    )
    save_img(
        best_y,
        f"results/{slurm_job_id}/best_rank{best_rank + 1}_y_false_color.png",
        false_color=True,
    )
    save_img(
        best_bicubic,
        f"results/{slurm_job_id}/best_rank{best_rank + 1}_bicubic_false_color.png",
        false_color=True,
    )
    save_img(
        model_best,
        f"results/{slurm_job_id}/best_rank{best_rank + 1}_model_false_color.png",
        false_color=True,
    )

    save_img_histogram(
        worst_x, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_x_histogram.png"
    )
    save_img_histogram(
        best_x, f"results/{slurm_job_id}/best_rank{best_rank + 1}_x_histogram.png"
    )
    save_img_histogram(
        worst_y, f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_y_histogram.png"
    )
    save_img_histogram(
        model_worse,
        f"results/{slurm_job_id}/worst_rank{worst_rank + 1}_model_histogram.png",
    )
    save_img_histogram(
        model_best,
        f"results/{slurm_job_id}/best_rank{best_rank + 1}_model_histogram.png",
    )
    save_img_histogram(
        best_y, f"results/{slurm_job_id}/best_rank{best_rank + 1}_y_histogram.png"
    )

    # Plot histogram of SSIM scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        bicubic, bins=30, alpha=0.7, label="Bicubic SSIM", color="blue", density=True
    )
    plt.hist(
        model_store, bins=30, alpha=0.7, label="Model SSIM", color="red", density=True
    )
    plt.xlabel("SSIM Score")
    plt.ylabel("Density")
    plt.title("Distribution of SSIM Scores: Bicubic vs Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f"results/{slurm_job_id}/ssim_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"Saved worst rank {worst_rank + 1} and best rank {best_rank + 1} images (regular and false color) to results/{slurm_job_id}/"
    )


if __name__ == "__main__":
    main()
