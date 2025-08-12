import argparse
import csv
import importlib as py_importlib
import importlib.util as importlib_util
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.metrics as skimetrics
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import init_dataloader
from models import Cond_VAE
from utils import save_img, save_img_histogram


def _config_label(g1: bool, rec: bool, g2: bool) -> str:
    """Short label for a (gamma_first, recurrent, gamma_second) config."""
    return f"g1-{int(g1)}\nrec-{int(rec)}\ng2-{int(g2)}"


def _plot_per_model_barchart(summary_rows: list[dict], out_dir: str, model_tag: str) -> None:
    """Create a per-model bar chart of SSIM means with error bars and bicubic band."""
    # Order rows in a stable config order: g1 in [0,1], rec in [0,1], g2 in [0,1]
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["gamma_first"], r["recurrent"], r["gamma_second"])  # type: ignore[index]
    )

    labels = [
        _config_label(bool(r["gamma_first"]), bool(r["recurrent"]), bool(r["gamma_second"]))  # type: ignore[index]
        for r in summary_rows_sorted
    ]
    means = [float(r["model_mean"]) for r in summary_rows_sorted]
    errs = [float(r.get("model_std", 0.0)) for r in summary_rows_sorted]

    # Bicubic reference (same across rows)
    bicubic_mean = float(summary_rows_sorted[0]["bicubic_mean"]) if summary_rows_sorted else 0.0
    bicubic_std = float(summary_rows_sorted[0].get("bicubic_std", 0.0)) if summary_rows_sorted else 0.0

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errs, capsize=4, color="#5DA5DA", alpha=0.9)
    plt.axhline(bicubic_mean, color="#F17CB0", linestyle="--", label=f"Bicubic mean = {bicubic_mean:.4f}")
    if bicubic_std > 0:
        plt.fill_between(
            [x[0] - 0.6, x[-1] + 0.6],
            [bicubic_mean - bicubic_std, bicubic_mean - bicubic_std],
            [bicubic_mean + bicubic_std, bicubic_mean + bicubic_std],
            color="#F17CB0",
            alpha=0.12,
            label="Bicubic ±1σ",
        )
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("SSIM mean ± std")
    plt.title(f"{model_tag}: SSIM across sampling configs")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "barchart.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_per_model_barchart_lpips(summary_rows: list[dict], out_dir: str, model_tag: str) -> None:
    """Create a per-model bar chart for LPIPS (lower is better)."""
    # If LPIPS not available, skip gracefully
    if not any("model_lpips_mean" in r for r in summary_rows):
        return
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["gamma_first"], r["recurrent"], r["gamma_second"])  # type: ignore[index]
    )
    labels = [
        _config_label(bool(r["gamma_first"]), bool(r["recurrent"]), bool(r["gamma_second"]))  # type: ignore[index]
        for r in summary_rows_sorted
    ]
    means = [float(r.get("model_lpips_mean", 0.0)) for r in summary_rows_sorted]
    errs = [float(r.get("model_lpips_std", 0.0)) for r in summary_rows_sorted]

    bicubic_mean = float(summary_rows_sorted[0].get("bicubic_lpips_mean", 0.0)) if summary_rows_sorted else 0.0
    bicubic_std = float(summary_rows_sorted[0].get("bicubic_lpips_std", 0.0)) if summary_rows_sorted else 0.0

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errs, capsize=4, color="#60BD68", alpha=0.9)
    plt.axhline(bicubic_mean, color="#F17CB0", linestyle="--", label=f"Bicubic LPIPS = {bicubic_mean:.4f}")
    if bicubic_std > 0:
        plt.fill_between(
            [x[0] - 0.6, x[-1] + 0.6],
            [bicubic_mean - bicubic_std, bicubic_mean - bicubic_std],
            [bicubic_mean + bicubic_std, bicubic_mean + bicubic_std],
            color="#F17CB0",
            alpha=0.12,
            label="Bicubic ±1σ",
        )
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("LPIPS mean ± std (lower is better)")
    plt.title(f"{model_tag}: LPIPS across sampling configs")
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "barchart_lpips.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_per_model_barchart_mmse(summary_rows: list[dict], out_dir: str, model_tag: str, metric: str = "ssim") -> None:
    """Create a per-model bar chart for MMSE using the given metric ('ssim', 'lpips' or 'psnr')."""
    assert metric in {"ssim", "lpips", "psnr"}
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["gamma_first"], r["recurrent"], r["gamma_second"])  # type: ignore[index]
    )
    labels = [
        _config_label(bool(r["gamma_first"]), bool(r["recurrent"]), bool(r["gamma_second"]))  # type: ignore[index]
        for r in summary_rows_sorted
    ]
    if metric == "ssim":
        means = [float(r.get("mmse_ssim_mean", 0.0)) for r in summary_rows_sorted]
        errs = [float(r.get("mmse_ssim_std", 0.0)) for r in summary_rows_sorted]
        ref_mean = float(summary_rows_sorted[0].get("bicubic_mean", 0.0)) if summary_rows_sorted else 0.0
        ref_std = float(summary_rows_sorted[0].get("bicubic_std", 0.0)) if summary_rows_sorted else 0.0
        ylabel = "MMSE SSIM mean ± std"
        title = f"{model_tag}: MMSE SSIM across sampling configs"
        color = "#5DA5DA"
    elif metric == "lpips":
        means = [float(r.get("mmse_lpips_mean", 0.0)) for r in summary_rows_sorted]
        errs = [float(r.get("mmse_lpips_std", 0.0)) for r in summary_rows_sorted]
        ref_mean = float(summary_rows_sorted[0].get("bicubic_lpips_mean", 0.0)) if summary_rows_sorted else 0.0
        ref_std = float(summary_rows_sorted[0].get("bicubic_lpips_std", 0.0)) if summary_rows_sorted else 0.0
        ylabel = "MMSE LPIPS mean ± std (lower is better)"
        title = f"{model_tag}: MMSE LPIPS across sampling configs"
        color = "#60BD68"
    else:
        means = [float(r.get("mmse_psnr_mean", 0.0)) for r in summary_rows_sorted]
        errs = [float(r.get("mmse_psnr_std", 0.0)) for r in summary_rows_sorted]
        ref_mean = float(summary_rows_sorted[0].get("bicubic_psnr_mean", 0.0)) if summary_rows_sorted else 0.0
        ref_std = float(summary_rows_sorted[0].get("bicubic_psnr_std", 0.0)) if summary_rows_sorted else 0.0
        ylabel = "MMSE PSNR (dB) mean ± std (higher is better)"
        title = f"{model_tag}: MMSE PSNR across sampling configs"
        color = "#FAA43A"

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errs, capsize=4, color=color, alpha=0.9)
    if metric == "ssim":
        plt.axhline(ref_mean, color="#F17CB0", linestyle="--", label=f"Bicubic SSIM = {ref_mean:.4f}")
    elif metric == "lpips":
        plt.axhline(ref_mean, color="#F17CB0", linestyle="--", label=f"Bicubic LPIPS = {ref_mean:.4f}")
    else:
        plt.axhline(ref_mean, color="#F17CB0", linestyle="--", label=f"Bicubic PSNR = {ref_mean:.2f} dB")
    if ref_std > 0:
        plt.fill_between(
            [x[0] - 0.6, x[-1] + 0.6],
            [ref_mean - ref_std, ref_mean - ref_std],
            [ref_mean + ref_std, ref_mean + ref_std],
            color="#F17CB0",
            alpha=0.12,
            label="Bicubic ±1σ",
        )
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    suffix = "mmse_ssim" if metric == "ssim" else ("mmse_lpips" if metric == "lpips" else "mmse_psnr")
    plt.savefig(os.path.join(out_dir, f"barchart_{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_per_model_barchart_psnr(summary_rows: list[dict], out_dir: str, model_tag: str) -> None:
    """Create a per-model bar chart for PSNR (higher is better)."""
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["gamma_first"], r["recurrent"], r["gamma_second"])  # type: ignore[index]
    )
    labels = [
        _config_label(bool(r["gamma_first"]), bool(r["recurrent"]), bool(r["gamma_second"]))  # type: ignore[index]
        for r in summary_rows_sorted
    ]
    means = [float(r.get("model_psnr_mean", 0.0)) for r in summary_rows_sorted]
    errs = [float(r.get("model_psnr_std", 0.0)) for r in summary_rows_sorted]

    bicubic_mean = float(summary_rows_sorted[0].get("bicubic_psnr_mean", 0.0)) if summary_rows_sorted else 0.0
    bicubic_std = float(summary_rows_sorted[0].get("bicubic_psnr_std", 0.0)) if summary_rows_sorted else 0.0

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errs, capsize=4, color="#FAA43A", alpha=0.9)
    plt.axhline(bicubic_mean, color="#F17CB0", linestyle="--", label=f"Bicubic PSNR = {bicubic_mean:.2f} dB")
    if bicubic_std > 0:
        plt.fill_between(
            [x[0] - 0.6, x[-1] + 0.6],
            [bicubic_mean - bicubic_std, bicubic_mean - bicubic_std],
            [bicubic_mean + bicubic_std, bicubic_mean + bicubic_std],
            color="#F17CB0",
            alpha=0.12,
            label="Bicubic ±1σ",
        )
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("PSNR (dB) mean ± std")
    plt.title(f"{model_tag}: PSNR across sampling configs")
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "barchart_psnr.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_global_barchart(results_root: str, model_summaries: list[tuple[str, str]]) -> None:
    """Create a global grouped bar chart across available models for each sampling config.

    model_summaries: list of tuples (model_tag, model_dir) where model_dir contains summary.csv
    """
    if not model_summaries:
        return

    # Read summaries
    per_model = {}
    configs = []
    for model_tag, model_dir in model_summaries:
        csv_path = os.path.join(model_dir, "summary.csv")
        if not os.path.exists(csv_path):
            continue
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Normalize ints
                r["gamma_first"] = int(r["gamma_first"]) if "gamma_first" in r else 0
                r["recurrent"] = int(r["recurrent"]) if "recurrent" in r else 0
                r["gamma_second"] = int(r["gamma_second"]) if "gamma_second" in r else 0
                rows.append(r)
        rows = sorted(rows, key=lambda r: (r["gamma_first"], r["recurrent"], r["gamma_second"]))
        per_model[model_tag] = rows
        if not configs and rows:
            configs = [
                (r["gamma_first"], r["recurrent"], r["gamma_second"]) for r in rows
            ]

    if not per_model:
        return

    # X positions per config
    n_configs = len(configs)
    model_tags = list(per_model.keys())
    n_models = len(model_tags)
    x = np.arange(n_configs)
    total_width = 0.8
    bar_width = total_width / max(n_models, 1)

    # Colors list
    colors = ["#5DA5DA", "#F15854", "#60BD68", "#FAA43A", "#B276B2", "#DECF3F"]

    plt.figure(figsize=(max(12, 3 * n_configs), 6))

    # Bicubic reference from first available model
    first_rows = next(iter(per_model.values()))
    bicubic_mean = float(first_rows[0].get("bicubic_mean", 0.0)) if first_rows else 0.0
    bicubic_std = float(first_rows[0].get("bicubic_std", 0.0)) if first_rows else 0.0
    plt.axhline(bicubic_mean, color="#7A68A6", linestyle="--", label=f"Bicubic mean = {bicubic_mean:.4f}")
    if bicubic_std > 0:
        plt.fill_between(
            [-0.6, n_configs - 1 + 0.6],
            [bicubic_mean - bicubic_std, bicubic_mean - bicubic_std],
            [bicubic_mean + bicubic_std, bicubic_mean + bicubic_std],
            color="#7A68A6",
            alpha=0.12,
            label="Bicubic ±1σ",
        )

    for m_idx, model_tag in enumerate(model_tags):
        rows = per_model[model_tag]
        means = [float(r.get("model_mean", 0.0)) for r in rows]
        errs = [float(r.get("model_std", 0.0)) for r in rows]
        positions = x - total_width / 2 + m_idx * bar_width + bar_width / 2
        plt.bar(positions, means, width=bar_width, yerr=errs, capsize=3,
                color=colors[m_idx % len(colors)], label=model_tag, alpha=0.9)

    labels = [_config_label(bool(g1), bool(rec), bool(g2)) for g1, rec, g2 in configs]
    plt.xticks(x, labels)
    plt.ylabel("SSIM mean ± std")
    plt.title("Global SSIM across models and sampling configs")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.legend(ncol=max(1, n_models))
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, "global_barchart.png"), dpi=300, bbox_inches="tight")
    plt.close()


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
            x_hat = model.sample_from_distribution(mean_decode, gamma, gamma_added_first).clamp(0, 1)
            # Recurrent pass through the full model
            x_hat, *_ = model.forward(x_hat, y_exp)
            # Optional second noise application
            x_hat = model.sample_from_distribution(x_hat, gamma, gamma_added_second)
            return x_hat.clamp(0,1)
        else:
            # No recurrent pass, just decide whether to add first noise or not
            x_hat = model.sample_from_distribution(mean_decode, gamma, gamma_added_first)
            return x_hat.clamp(0,1)


@torch.no_grad()
def compute_bicubic_ssim(val_loader, device) -> np.ndarray:
    """Compute and return bicubic SSIM array over the validation set (batch_size=1 expected)."""
    n = len(val_loader)
    scores = np.zeros(n, dtype=np.float32)
    for idx, (lr, hr) in enumerate(
        tqdm(val_loader, total=n, desc="Bicubic SSIM")
    ):
        hr = hr.to(device)
        lr = lr.to(device)
        up = F.interpolate(lr, scale_factor=2, mode="bicubic")
        scores[idx] = skimetrics.structural_similarity(
            hr[0].cpu().numpy(),
            up[0].cpu().numpy(),
            data_range=1.0,
            channel_axis=0,
        )
    return scores


@torch.no_grad()
def evaluate_config(model: Cond_VAE, val_loader, device, out_dir: str,
                    gamma_first: bool, recurrent: bool, gamma_second: bool,
                    bicubic_cache: np.ndarray | None,
                    index_to_log: int = 980,
                    samples_for_mmse: int = 50) -> dict:
    """
    Evaluate one (gamma_first, recurrent, gamma_second) config.
    Returns a dict with summary metrics. Saves logs for index_to_log into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    model.to(device)

    model_ssim = np.zeros(len(val_loader), dtype=np.float32)
    model_psnr = np.zeros(len(val_loader), dtype=np.float32)
    # Use provided bicubic cache or compute on the fly (first run should pass cache)
    if bicubic_cache is None:
        bicubic_cache = compute_bicubic_ssim(val_loader, device)

    # Prepare LPIPS loss (AlexNet backbone) as optional
    lpips_fn = None
    lpips_spec = importlib_util.find_spec("lpips")
    if lpips_spec is not None:
        try:
            lpips_module = py_importlib.import_module("lpips")
            LPIPS = lpips_module.LPIPS
            lpips_fn = LPIPS(net="alex").to(device)
            lpips_fn.eval()
        except Exception:
            lpips_fn = None

    # Containers for metrics
    mmse_ssim = np.zeros(len(val_loader), dtype=np.float32)
    mmse_psnr = np.zeros(len(val_loader), dtype=np.float32)
    # LPIPS arrays (only if available)
    bicubic_lp = np.zeros(len(val_loader), dtype=np.float32)
    model_lp = np.zeros(len(val_loader), dtype=np.float32)
    mmse_lp = np.zeros(len(val_loader), dtype=np.float32)

    # Evaluate over validation set
    for i, (y, x) in enumerate(
        tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Eval g1={int(gamma_first)} rec={int(recurrent)} g2={int(gamma_second)}",
        )
    ):
        x = x.to(device)
        y = y.to(device)
        out = sample_with_flags(
            model,
            y,
            samples=1,
            gamma_added_first=gamma_first,
            recurrent=recurrent,
            gamma_added_second=gamma_second,
        )
        # out shape: (1, C, H, W)
        model_ssim[i] = skimetrics.structural_similarity(
            out[0].cpu().numpy(),
            x[0].cpu().numpy(),
            data_range=1.0,
            channel_axis=0,
        )
        model_psnr[i] = float(
            skimetrics.peak_signal_noise_ratio(
                x[0].cpu().numpy(), out[0].cpu().numpy(), data_range=1.0
            )
        )

        # MMSE via multiple samples
        samples = sample_with_flags(
            model,
            y,
            samples=samples_for_mmse,
            gamma_added_first=gamma_first,
            recurrent=recurrent,
            gamma_added_second=gamma_second,
        )
        mmse_img = samples.mean(dim=0, keepdim=True)  # (1, C, H, W)
        mmse_ssim[i] = skimetrics.structural_similarity(
            mmse_img[0].cpu().numpy(),
            x[0].cpu().numpy(),
            data_range=1.0,
            channel_axis=0,
        )
        mmse_psnr[i] = float(
            skimetrics.peak_signal_noise_ratio(
                x[0].cpu().numpy(), mmse_img[0].cpu().numpy(), data_range=1.0
            )
        )

        # LPIPS computations (use bands [2,1,0] and range [-1,1])
        def to_lpips_3ch(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 3:
                t = t.unsqueeze(0)
            t3 = t[:, [2, 1, 0], :, :].clamp(0, 1)
            return (t3 * 2.0) - 1.0

        if lpips_fn is not None:
            bicubic_i = F.interpolate(y, scale_factor=2, mode="bicubic")  # (1, C, H, W)
            bicubic_lp[i] = float(lpips_fn(to_lpips_3ch(bicubic_i), to_lpips_3ch(x)).mean().item())
            model_lp[i] = float(lpips_fn(to_lpips_3ch(out), to_lpips_3ch(x)).mean().item())
            mmse_lp[i] = float(lpips_fn(to_lpips_3ch(mmse_img), to_lpips_3ch(x)).mean().item())
        # Specific logging for index_to_log
        if i == index_to_log:
            y_i, x_i = y[0].detach(), x[0].detach()
            out_i = out[0].detach()
            bicubic_i_vis = F.interpolate(y, scale_factor=2, mode="bicubic")[0].detach()

            save_img(x_i, os.path.join(out_dir, f"idx{index_to_log}_x.png"))
            save_img(y_i, os.path.join(out_dir, f"idx{index_to_log}_y.png"))
            save_img(bicubic_i_vis, os.path.join(out_dir, f"idx{index_to_log}_bicubic.png"))
            save_img(out_i, os.path.join(out_dir, f"idx{index_to_log}_model.png"))
            save_img(mmse_img[0].detach(), os.path.join(out_dir, f"idx{index_to_log}_mmse.png"))

            # False color versions
            save_img(x_i, os.path.join(out_dir, f"idx{index_to_log}_x_false_color.png"), false_color=True)
            save_img(y_i, os.path.join(out_dir, f"idx{index_to_log}_y_false_color.png"), false_color=True)
            save_img(bicubic_i_vis, os.path.join(out_dir, f"idx{index_to_log}_bicubic_false_color.png"), false_color=True)
            save_img(out_i, os.path.join(out_dir, f"idx{index_to_log}_model_false_color.png"), false_color=True)
            save_img(mmse_img[0].detach(), os.path.join(out_dir, f"idx{index_to_log}_mmse_false_color.png"), false_color=True)

            # Histograms
            save_img_histogram(x_i, os.path.join(out_dir, f"idx{index_to_log}_x_histogram.png"))
            save_img_histogram(y_i, os.path.join(out_dir, f"idx{index_to_log}_y_histogram.png"))
            save_img_histogram(out_i, os.path.join(out_dir, f"idx{index_to_log}_model_histogram.png"))
            save_img_histogram(mmse_img[0].detach(), os.path.join(out_dir, f"idx{index_to_log}_mmse_histogram.png"))

    # Compute bicubic PSNR once for reference
    bicubic_psnr = np.zeros(len(val_loader), dtype=np.float32)
    for j, (yy, xx) in enumerate(val_loader):
        xx = xx.to(device)
        yy = yy.to(device)
        up = F.interpolate(yy, scale_factor=2, mode="bicubic")
        bicubic_psnr[j] = float(
            skimetrics.peak_signal_noise_ratio(
                xx[0].cpu().numpy(), up[0].cpu().numpy(), data_range=1.0
            )
        )

    # Summary metrics
    metrics = {
        # SSIM
        "bicubic_mean": float(bicubic_cache.mean()),
        "bicubic_std": float(bicubic_cache.std(ddof=0)),
        "model_mean": float(model_ssim.mean()),
        "model_std": float(model_ssim.std(ddof=0)),
        "mmse_ssim_mean": float(mmse_ssim.mean()),
        "mmse_ssim_std": float(mmse_ssim.std(ddof=0)),
        "improvement": float(model_ssim.mean() - bicubic_cache.mean()),
        # PSNR (dB)
        "bicubic_psnr_mean": float(bicubic_psnr.mean()),
        "bicubic_psnr_std": float(bicubic_psnr.std(ddof=0)),
        "model_psnr_mean": float(model_psnr.mean()),
        "model_psnr_std": float(model_psnr.std(ddof=0)),
    "mmse_psnr_mean": float(mmse_psnr.mean()),
    "mmse_psnr_std": float(mmse_psnr.std(ddof=0)),
    }
    # Conditionally add LPIPS metrics
    if lpips_fn is not None:
        metrics.update({
            "bicubic_lpips_mean": float(bicubic_lp.mean()),
            "bicubic_lpips_std": float(bicubic_lp.std(ddof=0)),
            "model_lpips_mean": float(model_lp.mean()),
            "model_lpips_std": float(model_lp.std(ddof=0)),
            "mmse_lpips_mean": float(mmse_lp.mean()),
            "mmse_lpips_std": float(mmse_lp.std(ddof=0)),
        })

    # Save histogram for the SSIM distributions
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(0.5, 1.0, 100)  # fixed bins across runs
    plt.hist(bicubic_cache, bins=bin_edges, alpha=0.7, label="Bicubic SSIM", color="blue", density=True)
    plt.hist(model_ssim, bins=bin_edges, alpha=0.7, label="Model SSIM", color="red", density=True)
    plt.hist(mmse_ssim, bins=bin_edges, alpha=0.7, label="MMSE SSIM", color="green", density=True)
    plt.xlabel("SSIM Score")
    plt.ylabel("Density")
    plt.title("Distribution of SSIM Scores: Bicubic vs Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.savefig(os.path.join(out_dir, "ssim_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save histogram for the PSNR distributions (dB)
    plt.figure(figsize=(10, 6))
    min_psnr = float(min(bicubic_psnr.min(), model_psnr.min(), mmse_psnr.min()))
    max_psnr = float(max(bicubic_psnr.max(), model_psnr.max(), mmse_psnr.max()))
    psnr_bins = np.linspace(max(10.0, min_psnr), max(50.0, max_psnr), 100)
    plt.hist(bicubic_psnr, bins=psnr_bins, alpha=0.7, label="Bicubic PSNR", color="#7A68A6", density=True)
    plt.hist(model_psnr, bins=psnr_bins, alpha=0.7, label="Model PSNR", color="#FAA43A", density=True)
    plt.hist(mmse_psnr, bins=psnr_bins, alpha=0.7, label="MMSE PSNR", color="#60BD68", density=True)
    plt.xlabel("PSNR (dB)")
    plt.ylabel("Density")
    plt.title("Distribution of PSNR: Bicubic vs Model vs MMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "psnr_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save histogram for LPIPS distributions (lower is better) only if computed
    if lpips_fn is not None:
        plt.figure(figsize=(10, 6))
        lp_bins = np.linspace(0.0, max(1.0, float(max(bicubic_lp.max(), model_lp.max(), mmse_lp.max()))), 100)
        plt.hist(bicubic_lp, bins=lp_bins, alpha=0.7, label="Bicubic LPIPS", color="#7A68A6", density=True)
        plt.hist(model_lp, bins=lp_bins, alpha=0.7, label="Model LPIPS", color="#60BD68", density=True)
        plt.hist(mmse_lp, bins=lp_bins, alpha=0.7, label="MMSE LPIPS", color="#F15854", density=True)
        plt.xlabel("LPIPS (lower is better)")
        plt.ylabel("Density")
        plt.title("Distribution of LPIPS: Bicubic vs Model vs MMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "lpips_histogram.png"), dpi=300, bbox_inches="tight")
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma_first",
                "recurrent",
                "gamma_second",
                "bicubic_mean",
                "bicubic_std",
                "model_mean",
                "model_std",
                "mmse_ssim_mean",
                "mmse_ssim_std",
                "improvement",
                "bicubic_psnr_mean",
                "bicubic_psnr_std",
                "model_psnr_mean",
                "model_psnr_std",
                "bicubic_lpips_mean",
                "bicubic_lpips_std",
                "model_lpips_mean",
                "model_lpips_std",
                "mmse_lpips_mean",
                "mmse_lpips_std",
                "mmse_psnr_mean",
                "mmse_psnr_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {csv_path}")

    # Create per-model barchart
    try:
        _plot_per_model_barchart(summary_rows, base_out_dir, model_tag)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Warning: failed to create per-model barchart for {model_tag}: {e}")
    # Create LPIPS barchart
    try:
        _plot_per_model_barchart_lpips(summary_rows, base_out_dir, model_tag)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Warning: failed to create per-model LPIPS barchart for {model_tag}: {e}")
    # Create PSNR barchart
    try:
        _plot_per_model_barchart_psnr(summary_rows, base_out_dir, model_tag)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Warning: failed to create per-model PSNR barchart for {model_tag}: {e}")
    # Create MMSE barcharts
    try:
        _plot_per_model_barchart_mmse(summary_rows, base_out_dir, model_tag, metric="ssim")
        _plot_per_model_barchart_mmse(summary_rows, base_out_dir, model_tag, metric="lpips")
        _plot_per_model_barchart_mmse(summary_rows, base_out_dir, model_tag, metric="psnr")
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Warning: failed to create per-model MMSE barcharts for {model_tag}: {e}")


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
    model_summaries: list[tuple[str, str]] = []
    if args.laplacian_ckpt is not None and os.path.exists(args.laplacian_ckpt):
        lap_model = load_model(
            args.laplacian_ckpt, distribution_type="laplacian", device=device,
            cr=args.cr, patch_size=args.patch_size
        )
        lap_tag = f"Laplacian-{os.path.basename(args.laplacian_ckpt).split('.')[0]}"
        lap_out_dir = os.path.join(results_root, lap_tag)
        os.makedirs(lap_out_dir, exist_ok=True)
        evaluate_model_grid(lap_model, lap_tag, val_loader, device, lap_out_dir, args.index_to_log, bicubic_cache)
        model_summaries.append((lap_tag, lap_out_dir))
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
            model_summaries.append((gau_tag, gau_out_dir))
        else:
            print(f"Gaussian checkpoint not found: {args.gaussian_ckpt}. Skipping Gaussian model.")

    # Global barchart across models/configs
    try:
        _plot_global_barchart(results_root, model_summaries)
        print(f"Saved global barchart to {os.path.join(results_root, 'global_barchart.png')}")
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Warning: failed to create global barchart: {e}")

    print("All evaluations complete.")


if __name__ == "__main__":
    main()
