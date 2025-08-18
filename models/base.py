import abc
import os
from math import isnan
from typing import List

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from skimage import metrics as skmetrics
from tqdm import tqdm

from callbacks import Callback
from utils import icp, quantile_map


class BaseVAE(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for all VAEs. Defines the common interface for training and validation.
    """

    def __init__(
        self,
        patch_size: int = 64,
        callbacks: List[Callback] | None = None,
        slurm_job_id: str = "local",
    ):
        if callbacks is None:
            callbacks = []
        super().__init__()
        # Scheduler to reduce learning rate on plateau
        self.latent_size: int = 0
        self.slurm_job_id: str = slurm_job_id
        self.patch_size: int = patch_size
        self.callbacks: List[Callback] = callbacks
        self.ssim = skmetrics.structural_similarity
        self.lpips_fn = lpips.LPIPS(net="alex")
        self.num_params: int = 0

    def fit(self, train_loader, val_loader, device, optimizer, epochs=1000, **kwargs):
        """
        Fit the model to the training data.
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: device to use for training (e.g., 'cuda' or 'cpu')
            epochs: number of epochs to train
        """
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500
        )
        self.current_epoch: int = 0
        self.lpips_fn = self.lpips_fn.to(device)
        self.lpips_fn.eval()
        start_epoch = kwargs.get("start_epoch", 1)
        val_metrics_every = kwargs.get("val_metrics_every", float("inf"))
        x, _ = next(iter(train_loader))
        b = x.size(0)

        self.wandb_run = wandb.init(
            project=self.__class__.__name__,
            name=f"Latent-{self.latent_size}-Patch-{self.patch_size}-SLURM-{kwargs.get('slurm_job_id', 'local')}",
            entity="ebardet-isae-supaero",
            config=kwargs.get(
                "config",
                {
                    "latent_size": self.latent_size,
                    "patch_size": self.patch_size,
                    "epochs": epochs,
                    "batch_size": b,
                    "val_metrics_every": val_metrics_every,
                    "slurm_job_id": kwargs.get("slurm_job_id", "local"),
                    "Parameter_number": self.num_params,
                    "cr": self.cr,
                    "L": kwargs.get("L", None),
                    "gamma_type": self.gamma_type
                    if hasattr(self, "gamma_type")
                    else "scalar",
                    "distribution_type": self.distribution_type
                    if hasattr(self, "distribution_type")
                    else "gaussian",
                },
            ),
        )

        optimizer = self.optimizer
        self.on_train_start()

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            for cb in self.callbacks:
                if cb.on_epoch_begin(
                    epoch=epoch, optimizer=optimizer, device=device, model=self
                ):
                    print(
                        f"Stopping training before epoch {epoch} due to {cb.__class__.__name__} condition."
                    )
                    return  # Stop training if callback indicates to stop
            self.train()
            train_loss = 0.0
            terms_dict = {}
            for _, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training, Epoch {epoch}/{epochs}",
                unit="batch",
            ):
                optimizer.zero_grad()
                loss, terms = self.train_step(batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                if not terms_dict:
                    terms_dict = terms
                else:
                    for key, value in terms.items():
                        if key in terms_dict:
                            terms_dict[key] += value
                        else:
                            terms_dict[key] = value
                train_loss += loss.item()

            # Average the loss terms
            for key in terms_dict:
                terms_dict[key] /= len(train_loader) * b
            self.terms_dict = terms_dict
            train_loss /= len(train_loader) * b
            self.log(self.wandb_run, terms_dict, step=epoch)

            if isnan(train_loss):
                raise ValueError(
                    f"NaN detected in training loss at epoch {epoch}. Check your model and data."
                )

            self.on_train_epoch_end()
            self.eval()
            val_loss = 0.0
            val_terms_dict = {}
            with torch.no_grad():
                for _, batch in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Validation, Epoch {epoch}/{epochs}",
                    unit="batch",
                ):
                    loss, terms = self.val_step(batch, device)
                    if not val_terms_dict:
                        val_terms_dict = terms
                    else:
                        for key, value in terms.items():
                            if key in val_terms_dict:
                                val_terms_dict[key] += value
                            else:
                                val_terms_dict[key] = value

                    val_loss += loss.item()

                if epoch % val_metrics_every == 0 or epoch in [1, epochs]:
                    full_val = True
                else:
                    full_val = False
                self.evaluate(val_loader, self.wandb_run, epoch, full_val=full_val)

            # Average the validation loss terms
            for key in val_terms_dict:
                val_terms_dict[key] /= len(val_loader) * b

            val_loss /= len(val_loader) * b
            if self.scheduler:
                self.scheduler.step(val_loss)
            self.log(self.wandb_run, val_terms_dict, step=epoch)
            for cb in self.callbacks:
                if cb.on_epoch_end(
                    epoch=epoch,
                    optimizer=optimizer,
                    device=device,
                    model=self,
                    logs=val_terms_dict,
                ):
                    print(
                        f"Stopping training after epoch {epoch} due to {cb.__class__.__name__} condition."
                    )
                    return  # Stop training if callback indicates to stop

            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        return

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for the forward pass of the model.
        """
        raise NotImplementedError("forward must be implemented in the derived class.")

    @abc.abstractmethod
    def train_step(self, batch, device):
        """
        Performs a training step on a batch.
        Args:
            batch: data batch
            device: device to use
        Returns:
            loss: scalar loss value
            logs (dict): dictionary of values to log, e.g., {'loss': loss_value, 'kld': kld_value}
        """
        raise NotImplementedError(
            "train_step must be implemented in the derived class."
        )

    @abc.abstractmethod
    def val_step(self, batch, device):
        """
        Performs a validation step on a batch.
        Args:
            batch: data batch
            device: device to use
            loss_fn: loss function
        Returns:
            loss: scalar loss value
            logs (dict): dictionary of values to log, e.g., {'val_loss': loss_value, 'val_kld': kld_value}
        """
        raise NotImplementedError("val_step must be implemented in the derived class.")

    @abc.abstractmethod
    def evaluate(self, val_loader, wandb_run, epoch, full_val):
        """
        Evaluate the model on the validation set.
        Args:
            val_loader: DataLoader for validation data
            wandb_run: wandb run instance for logging
            epoch: current epoch number
            full_val: whether to compute full validation metrics or just log part of it
        """
        raise NotImplementedError("evaluate must be implemented in the derived class.")

    def log(self, wandb_run, logs: dict, step=None):
        """
        Optional method to log model-specific information.
        Args:
            wandb_run: wandb run instance for logging
            logs: dictionary of values to log
            step: current step (optional)
        """
        if not wandb_run:
            print("WandB run not initialized, skipping logging.")
            return
        if step is not None:
            wandb_run.log(logs, step=step)

    @abc.abstractmethod
    def on_train_start(self, **kwargs):
        """
        Called at the start of training.
        Args:
            kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def on_train_epoch_end(self, **kwargs):
        """
        Called at the end of each training epoch.
        Args:
            kwargs: additional arguments
        """
        pass

    @abc.abstractmethod
    def sample(self, y, samples=1000):
        """
        Sample from the model given input y.
        Args:
            y: input data
            samples: number of samples to generate
        Returns:
            samples: generated samples
        """
        raise NotImplementedError("sample must be implemented in the derived class.")

    @abc.abstractmethod
    def get_task_data(self, val_loader):
        """
        Get the data for the task.
        Args:
            val_loader: DataLoader for validation data
        Returns:
            pred: predicted data
            target: target data
        """
        raise NotImplementedError(
            "get_task_data must be implemented in the derived class."
        )

    def task(self, val_loader):
        """
        Performs a test step on a batch.
        Args:
            batch: data batch
            device: device to use
        """

        results_dir = os.path.join("results", f"{self.slurm_job_id}_CRx{self.cr}")
        os.makedirs(results_dir, exist_ok=True)

        pred, target = self.get_task_data(val_loader)

        with torch.no_grad():
            samples = self.sample(pred, gamma_added=True, recurrent=False)[
                :, 0, :, :, :
            ]
            if not hasattr(self, "gamma"):
                _, _, _, self.gamma = self.forward(target, pred)
            if self.gamma_type == "scalar":
                gamma = self.gamma
            else:
                gamma = self.gamma[0, :, :, :].mean(0)
                normed_gamma = (gamma - gamma.min()) / (gamma.max() - gamma.min())
        # save_img_histogram(pred, f"{results_dir}/input_image_histogram.png")
        # save_img_histogram(target, f"{results_dir}/target_image_histogram.png")
        mean = samples.mean(dim=0)
        std = samples.std(dim=0).cpu().numpy().mean(axis=0)
        pred_bicubic = nn.functional.interpolate(
            pred, scale_factor=2, mode="bicubic", align_corners=False
        )

        quantiles = torch.arange(
            0,
            1.05,
            0.05,
        ).to(pred.device)
        empirical_coverage = icp(samples, target, quantiles)

        quantile_90 = quantile_map(samples, 0.9)
        plt.figure()
        plt.plot(quantiles.cpu().numpy(), empirical_coverage.cpu().numpy(), marker="o")
        plt.xlabel("Target Coverage")
        plt.ylabel("True Coverage")
        plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Ideal Coverage")
        plt.title("Empirical vs Target Coverage")
        plt.legend()
        plt.savefig(f"{results_dir}/empirical_coverage.png", bbox_inches="tight")
        plt.close()

        plt.imsave(
            f"{results_dir}/input_image.png",
            pred_bicubic[0, [2, 1, 0], :, :]
            .clip(0, 1)
            .cpu()
            .numpy()
            .transpose(1, 2, 0),
        )
        plt.imsave(
            f"{results_dir}/sampled_image.png",
            samples[0, [2, 1, 0], :, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0),
        )
        plt.imsave(
            f"{results_dir}/ground_truth_image.png",
            target[0, [2, 1, 0], :, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0),
        )
        plt.imsave(
            f"{results_dir}/mean_of_samples.png",
            mean[[2, 1, 0], :, :].clip(0, 1).cpu().numpy().transpose(1, 2, 0),
        )

        mean_bias = (target - samples.mean(dim=0))[0].mean(dim=0).cpu().numpy()
        lim = max(abs(mean_bias.min()), abs(mean_bias.max()))

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean bias plot
        im1 = axes[0, 0].imshow(mean_bias, cmap="twilight", vmin=-lim, vmax=lim)
        axes[0, 0].set_title("Mean Bias")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # Quantile 90 plot
        im2 = axes[0, 1].imshow(quantile_90.squeeze().cpu().numpy(), cmap="viridis")
        axes[0, 1].set_title("Quantile 90")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # Standard deviation plot
        im3 = axes[1, 0].imshow(std, cmap="hot")
        axes[1, 0].set_title("Standard Deviation")
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0])

        # Gamma map plot (if available)
        if "gamma" in locals() and self.gamma_type != "scalar":
            im4 = axes[1, 1].imshow(normed_gamma.cpu().numpy(), cmap="hot")
            axes[1, 1].set_title("Gamma Map")
            axes[1, 1].axis("off")
            plt.colorbar(im4, ax=axes[1, 1])
        else:
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{results_dir}/error_mean_std_maps.png", bbox_inches="tight", dpi=150
        )
        plt.close()

        normed_std = (std - std.min()) / (std.max() - std.min())
        abs_bias = abs(mean_bias)
        normed_bias = (abs_bias - abs_bias.min()) / (abs_bias.max() - abs_bias.min())

        xor = normed_std + normed_bias - 2 * normed_std * normed_bias
        xor = np.clip(xor, 0, 1)  # Ensure values are between 0 and 1
        plt.imsave(f"{results_dir}/xor_mean_bias_std.png", xor, cmap="viridis")

        SSIM_MMSE = self.ssim(
            mean[:, :, :].cpu().numpy(),
            target[0, :, :, :].cpu().numpy(),
            data_range=1.0,
            multichannel=True,
            channel_axis=0,
        )
        print(f"SSIM MMSE: {SSIM_MMSE:.4f}")

        SSIM_samples = self.ssim(
            samples[0, :, :, :].cpu().numpy(),
            target[0, :, :, :].cpu().numpy(),
            data_range=1.0,
            multichannel=True,
            channel_axis=0,
        )
        print(f"SSIM Samples: {SSIM_samples:.4f}")

        SSIM_bicubic = self.ssim(
            pred_bicubic[0, :, :, :].cpu().numpy(),
            target[0, :, :, :].cpu().numpy(),
            data_range=1.0,
            multichannel=True,
            channel_axis=0,
        )
        print(f"SSIM Bicubic: {SSIM_bicubic:.4f}")

        E_MMSE_GT = (mean - target[0]).pow(2).mean()
        print(f"E_MMSE-GT: {E_MMSE_GT:.4f}")

        E_samples_GT = (samples - target).pow(2).mean()
        print(f"E_samples-GT: {E_samples_GT:.4f}")

        E_bicubic_GT = (pred_bicubic - target).pow(2).mean()
        print(f"E_bicubic-GT: {E_bicubic_GT:.4f}")

        if hasattr(self, "wandb_run") and self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "Metrics/MMSE": E_samples_GT,
                    "Plots/Error Maps": wandb.Image(
                        f"{results_dir}/error_mean_std_maps.png"
                    ),
                },
            )

            self.wandb_run.finish()

        else:
            print("WandB run not initialized, skipping logging.")

        print(f"Results saved to {results_dir}")
        return results_dir
