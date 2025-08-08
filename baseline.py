import os
import argparse
import json
import csv

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import init_dataloader
from models import Cond_VAE
from utils import save_img, save_img_histogram


def sample_with_flags(model: Cond_VAE, y: torch.Tensor, samples: int,
                      gamma_added_first: bool, recurrent: bool, gamma_added_second: bool) -> torch.Tensor:
    """
    Custom sampling to control noise before and after recurrent pass independently.
    Returns tensor shaped (samples, C, H, W).
    """
    device = y.device
    with torch.no_grad():
        mu, logvar = model.cond_prior(y).chunk(2, dim=1)
        mu, logvar = model.conv_condmu(mu), model.conv_condlogvar(logvar)
        _, _, h, w = y.shape
        z = torch.randn(
            samples, int(int(256 / (model.adjust * 2)) * 2 * h * w / 64), device=device
        )
        z = mu + torch.exp(0.5 * logvar) * z
        gamma = model.decoder_variance(z)
        # Store gamma on model for compatibility with downstream utils
        model.gamma = gamma
        if y.shape[0] == 1:
            y_exp = y.expand(samples, -1, -1, -1)
        else:
            y_exp = y
        mean_decode = model.decode(z, y_exp)

        if recurrent:
            x_hat = model.sample_from_distribution(mean_decode, gamma, gamma_added_first)
            # Recurrent pass through the full model
            x_hat, *_ = model.forward(x_hat, y_exp)
            # Optional second noise application
            x_hat = model.sample_from_distribution(x_hat, gamma, gamma_added_second)
            return x_hat
        else:
            # No recurrent pass, just decide whether to add first noise or not
            x_hat = model.sample_from_distribution(mean_decode, gamma, gamma_added_first)
            return x_hat


@torch.no_grad()
def compute_bicubic_ssim(val_loader, device) -> np.ndarray:
    """Compute and return bicubic SSIM array over the validation set (batch_size=1 expected)."""
    bicubic = np.zeros(len(val_loader), dtype=np.float32)
    for i, (y, x) in enumerate(tqdm(val_loader, total=len(val_loader), desc="Bicubic SSIM")):
        x = x.to(device)
        y = y.to(device)
        bicubic_upsampled = F.interpolate(y, scale_factor=2, mode="bicubic")
        bicubic[i] = ssim(
            x[0].cpu().numpy(),
            bicubic_upsampled[0].cpu().numpy(),
            data_range=1.0,
            channel_axis=0,
        )
    return bicubic


@torch.no_grad()
def evaluate_config(model: Cond_VAE, val_loader, device, out_dir: str,
                    gamma_first: bool, recurrent: bool, gamma_second: bool,
                    bicubic_cache: np.ndarray | None,
                    index_to_log: int = 980) -> dict:
    """
    Evaluate one (gamma_first, recurrent, gamma_second) config.
    Returns a dict with summary metrics. Saves logs for index_to_log into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    model.to(device)

    model_ssim = np.zeros(len(val_loader), dtype=np.float32)
    # Use provided bicubic cache or compute on the fly (first run should pass cache)
    if bicubic_cache is None:
        bicubic_cache = compute_bicubic_ssim(val_loader, device)

    # Evaluate over validation set
    for i, (y, x) in enumerate(tqdm(val_loader, total=len(val_loader), desc=f"Eval g1={int(gamma_first)} rec={int(recurrent)} g2={int(gamma_second)}")):
        x = x.to(device)
        y = y.to(device)
        out = sample_with_flags(model, y, samples=1,
                                gamma_added_first=gamma_first,
                                recurrent=recurrent,
                                gamma_added_second=gamma_second)
        # out shape: (1, C, H, W)
        model_ssim[i] = ssim(
            out[0].cpu().numpy(),
            x[0].cpu().numpy(),
            data_range=1.0,
            channel_axis=0,
        )

        # Specific logging for index_to_log
        if i == index_to_log:
            y_i, x_i = y[0].detach(), x[0].detach()
            out_i = out[0].detach()
            bicubic_i = F.interpolate(y, scale_factor=2, mode="bicubic")[0].detach()

            save_img(x_i, os.path.join(out_dir, f"idx{index_to_log}_x.png"))
            save_img(y_i, os.path.join(out_dir, f"idx{index_to_log}_y.png"))
            save_img(bicubic_i, os.path.join(out_dir, f"idx{index_to_log}_bicubic.png"))
            save_img(out_i, os.path.join(out_dir, f"idx{index_to_log}_model.png"))

            # False color versions
            save_img(x_i, os.path.join(out_dir, f"idx{index_to_log}_x_false_color.png"), false_color=True)
            save_img(y_i, os.path.join(out_dir, f"idx{index_to_log}_y_false_color.png"), false_color=True)
            save_img(bicubic_i, os.path.join(out_dir, f"idx{index_to_log}_bicubic_false_color.png"), false_color=True)
            save_img(out_i, os.path.join(out_dir, f"idx{index_to_log}_model_false_color.png"), false_color=True)

            # Histograms
            save_img_histogram(x_i, os.path.join(out_dir, f"idx{index_to_log}_x_histogram.png"))
            save_img_histogram(y_i, os.path.join(out_dir, f"idx{index_to_log}_y_histogram.png"))
            save_img_histogram(out_i, os.path.join(out_dir, f"idx{index_to_log}_model_histogram.png"))

    # Summary metrics
    metrics = {
        "bicubic_mean": float(bicubic_cache.mean()),
        "model_mean": float(model_ssim.mean()),
        "improvement": float(model_ssim.mean() - bicubic_cache.mean()),
    }

    # Save histogram for the SSIM distributions
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(0.0, 1.0, 31)  # fixed bins across runs
    plt.hist(bicubic_cache, bins=bin_edges, alpha=0.7, label="Bicubic SSIM", color="blue", density=True)
    plt.hist(model_ssim, bins=bin_edges, alpha=0.7, label="Model SSIM", color="red", density=True)
    plt.xlabel("SSIM Score")
    plt.ylabel("Density")
    plt.title("Distribution of SSIM Scores: Bicubic vs Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.savefig(os.path.join(out_dir, "ssim_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save metrics JSON
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model(ckpt_path: str, distribution_type: str, device: torch.device,
               cr: float = 1.5, patch_size: int = 256, gamma_type: str = "scalar") -> Cond_VAE:
    model = Cond_VAE(
        cr=cr,
        patch_size=patch_size,
        gamma_type=gamma_type,
        slurm_job_id=os.getenv("SLURM_JOB_ID", "local_run"),
        distribution_type=distribution_type,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])  # support pl-style
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def evaluate_model_grid(model: Cond_VAE, model_tag: str, val_loader, device,
                        base_out_dir: str, index_to_log: int = 980,
                        bicubic_cache: np.ndarray | None = None) -> None:
    """
    Run the full 2x2x2 grid for a given model and save a CSV summary.
    """
    combos = []
    for g1 in [False, True]:
        for rec in [False, True]:
            for g2 in [False, True]:
                combos.append((g1, rec, g2))

    summary_rows = []
    for g1, rec, g2 in combos:
        cfg_dir = os.path.join(base_out_dir, f"g1-{int(g1)}_rec-{int(rec)}_g2-{int(g2)}")
        metrics = evaluate_config(
            model,
            val_loader,
            device,
            cfg_dir,
            gamma_first=g1,
            recurrent=rec,
            gamma_second=g2,
            bicubic_cache=bicubic_cache,
            index_to_log=index_to_log,
        )
        summary_rows.append({
            "gamma_first": int(g1),
            "recurrent": int(rec),
            "gamma_second": int(g2),
            **metrics,
        })

    # Write CSV summary
    csv_path = os.path.join(base_out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma_first", "recurrent", "gamma_second", "bicubic_mean", "model_mean", "improvement"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hyperparameter influence grid for Cond_VAE sampling")
    parser.add_argument("--laplacian_ckpt", type=str, default="ckpt/3871425.pth", help="Path to Laplacian model checkpoint")
    parser.add_argument("--gaussian_ckpt", type=str, default="ckpt/3871424.pth", help="Path to Gaussian model checkpoint (optional)")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--cr", type=float, default=1.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--index_to_log", type=int, default=980)
    return parser.parse_args()


def main():
    args = parse_args()
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local_run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    _, val_loader = init_dataloader("s2v", batch_size=args.batch_size, patch_size=args.patch_size)

    # Compute bicubic SSIM once for the dataset
    print("Computing bicubic SSIM baseline once across validation set...")
    bicubic_cache = compute_bicubic_ssim(val_loader, device)
    print(f"Bicubic SSIM: {bicubic_cache.mean().item()}")

    results_root = os.path.join("results", slurm_job_id)
    os.makedirs(results_root, exist_ok=True)

    # Evaluate Laplacian model
    if args.laplacian_ckpt is not None and os.path.exists(args.laplacian_ckpt):
        lap_model = load_model(
            args.laplacian_ckpt, distribution_type="laplacian", device=device,
            cr=args.cr, patch_size=args.patch_size
        )
        lap_tag = f"Laplacian-{os.path.basename(args.laplacian_ckpt).split('.')[0]}"
        lap_out_dir = os.path.join(results_root, lap_tag)
        os.makedirs(lap_out_dir, exist_ok=True)
        evaluate_model_grid(lap_model, lap_tag, val_loader, device, lap_out_dir, args.index_to_log, bicubic_cache)
    else:
        print("No Laplacian checkpoint provided or path not found; skipping Laplacian model.")

    # Evaluate Gaussian model (optional)
    if args.gaussian_ckpt is not None:
        if os.path.exists(args.gaussian_ckpt):
            gau_model = load_model(
                args.gaussian_ckpt, distribution_type="gaussian", device=device,
                cr=args.cr, patch_size=args.patch_size
            )
            gau_tag = f"Gaussian-{os.path.basename(args.gaussian_ckpt).split('.')[0]}"
            gau_out_dir = os.path.join(results_root, gau_tag)
            os.makedirs(gau_out_dir, exist_ok=True)
            evaluate_model_grid(gau_model, gau_tag, val_loader, device, gau_out_dir, args.index_to_log, bicubic_cache)
        else:
            print(f"Gaussian checkpoint not found: {args.gaussian_ckpt}. Skipping Gaussian model.")

    print("All evaluations complete.")


if __name__ == "__main__":
    main()
