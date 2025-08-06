import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from loss import cond_loss
from utils import laplacian_sampling

from .base import BaseVAE
from .layers import conv_block, down_block, up_block


class Cond_VAE(BaseVAE):
    """
    Variational Autoencoder (VAE) model for image reconstruction.
    Inherits from BaseVAE and implements the VAE architecture with an encoder,
    reparameterization trick, and decoder. It also includes methods for training,
    validation, and evaluation.

    Args:
        latent_size (int): Size of the latent space.
        patch_size (int): Size of the input image patches (default: 64).
        callbacks (list, optional): List of callbacks to use during training (default: None).
    """

    def __init__(
        self,
        cr,
        patch_size=64,
        callbacks=None,
        slurm_job_id="local",
        L=1,
        gamma_type="scalar",
    ):
        if callbacks is None:
            callbacks = []
        super(Cond_VAE, self).__init__(patch_size, callbacks, slurm_job_id)
        self.cr = cr
        self.L: int = L  # Number of samples to draw from the latent space
        self.adjust = cr / 4  # Ensure latent size is a multiple of 4
        self.patch_size = patch_size
        self.gamma_type = gamma_type

        self.cond_prior = nn.Sequential(
            down_block(in_channels=4, out_channels=64),  # out 16 , 16 , 16
            down_block(in_channels=64, out_channels=128),  # out 64, 8, 8
            down_block(128, 256),
            conv_block(
                256,
                int(256 / (2 * self.adjust)) * 4,
                1,
                1,
                0,
                final_relu=False,
            ),
            # out 512 * 2 * 2 = 2048
        )

        self.hr_down = down_block(in_channels=4, out_channels=32)  # out 16 , 16 , 16

        self.encoder = nn.Sequential(
            down_block(in_channels=36, out_channels=64),  # out 64, 8, 8
            down_block(64, 128),
            down_block(128, 256),
            conv_block(
                256,
                int(256 / (2 * self.adjust)) * 4,
                1,
                1,
                0,
                final_relu=False,
            ),
            # out 512 * 2 * 2 = 2048
        )

        self.conv_mu = nn.Sequential(
            conv_block(
                int(256 / (2 * self.adjust)) * 2,
                int(256 / (2 * self.adjust)) * 2,
                3,
                1,
                1,
            ),
            nn.Flatten(start_dim=1),
        )
        self.conv_logvar = nn.Sequential(
            conv_block(
                int(256 / (2 * self.adjust)) * 2,
                int(256 / (2 * self.adjust)) * 2,
                3,
                1,
                1,
            ),
            nn.Flatten(start_dim=1),
        )
        self.conv_condmu = nn.Sequential(
            conv_block(
                int(256 / (2 * self.adjust)) * 2,
                int(256 / (2 * self.adjust)) * 2,
                3,
                1,
                1,
            ),
            nn.Flatten(start_dim=1),
        )
        self.conv_condlogvar = nn.Sequential(
            conv_block(
                int(256 / (2 * self.adjust)) * 2,
                int(256 / (2 * self.adjust)) * 2,
                3,
                1,
                1,
            ),
            nn.Flatten(start_dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    int(256 / (2 * self.adjust)) * 2,
                    patch_size // 2**4,
                    patch_size // 2**4,
                ),
            ),
            up_block(
                in_channels=int(256 / (2 * self.adjust)) * 2,
                out_channels=512,
            ),
            up_block(
                in_channels=512,
                out_channels=256,
            ),
            conv_block(256, 128, 3, 1, 1),
            conv_block(128, 64, 3, 1, 1),
            up_block(64, 32),
        )
        self.decoder_end = nn.Sequential(
            up_block(in_channels=36, out_channels=16),  # upsample to 8x8
            conv_block(16, 8, 3, 1, 1),
            conv_block(8, 8, 3, 1, 1),
            conv_block(
                8, 4, 3, 1, 1, final_relu=False
            ),  # Final conv to match input channels
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )

        if gamma_type == "scalar":
            self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.variance_decoder = nn.Sequential(
                nn.Unflatten(
                    1,
                    (
                        int(256 / (2 * self.adjust)) * 2,
                        patch_size // 2**4,
                        patch_size // 2**4,
                    ),
                ),
                up_block(
                    in_channels=int(256 / (2 * self.adjust)) * 2,
                    out_channels=256,
                ),
                up_block(
                    in_channels=256,
                    out_channels=128,
                ),
                up_block(
                    in_channels=128,
                    out_channels=64,
                ),
                up_block(64, 32),
                conv_block(32, 16, 1, 1, 0),
                conv_block(16, 8, 1, 1, 0),
                conv_block(
                    8, 4, 1, 1, 0, final_relu=False
                ),  # Final conv to match input channels
                nn.Sigmoid(),  # Ensure output is in [0, 1]
            )

        # 4 output channels (same as input)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        # Define the encoder part of the VAE
        x_encoded = self.hr_down(x)
        stack = torch.cat((x_encoded, y), dim=1)  # Concatenate with condition y
        pre_mu, pre_logvar = self.encoder(stack).chunk(
            2, dim=1
        )  # Split into mu and logvar
        return self.conv_mu(pre_mu), self.conv_logvar(pre_logvar)

    def reparameterize(self, mu, logvar, L):
        """
        Reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.
            L (int): Number of samples to draw from the latent space.
        Returns:
            torch.Tensor: Sampled latent vector of size (L, Batch, Latent_size).
        """
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        if L > 1:
            mu = mu.unsqueeze(0).expand(
                L, -1, -1
            )  # Expand mu to (L, Batch, Latent_size)
            std = std.unsqueeze(0).expand(L, -1, -1)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y) -> torch.Tensor:
        z_decoded = self.decoder(z)
        cat = torch.cat((z_decoded, y), dim=1)  # Concatenate with original input
        return self.decoder_end(cat)

    def decoder_variance(self, z) -> torch.Tensor:
        if self.gamma_type == "scalar":
            return self.gamma
        else:
            return self.variance_decoder(z)

    def forward(self, x, y, L=1):
        # Forward pass through the VAE
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar, L)
        if z.ndim == 3:
            decoded = []
            for i in range(L):
                decoded.append(self.decode(z[i], y))
                return (
                    torch.stack(decoded, dim=0),
                    mu,
                    logvar,
                    self.decoder_variance(z.mean(dim=0)),
                )

        return self.decode(z, y), mu, logvar, self.decoder_variance(z)

    def train_step(self, batch, device):
        y, x = batch
        x, y = x.to(device), y.to(device)
        x_hat, mu, logvar, gamma = self.forward(x, y, self.L)
        self.gamma = gamma
        cond_mu, cond_logvar = self.cond_prior(y).chunk(
            2, dim=1
        )  # Split into mu and logvar
        cond_mu, cond_logvar = (
            self.conv_condmu(cond_mu),
            self.conv_condlogvar(cond_logvar),
        )
        mse, kld = cond_loss(
            x_hat,
            x,
            mu,
            logvar,
            cond_mu,
            cond_logvar,
            self.gamma,
        )
        loss = mse + kld
        logs = {
            "Loss/loss": loss.item(),
            "Loss/mse": mse.item(),
            "Loss/kld": kld.item(),
        }
        return loss, logs

    def val_step(self, batch, device):
        y, x = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_hat, mu, logvar, gamma = self.forward(x, y)
            self.gamma = gamma
            cond_mu, cond_logvar = self.cond_prior(y).chunk(
                2, dim=1
            )  # Split into mu and logvar
            cond_mu, cond_logvar = (
                self.conv_condmu(cond_mu),
                self.conv_condlogvar(cond_logvar),
            )
            mse, kld = cond_loss(
                x_hat,
                x,
                mu,
                logvar,
                cond_mu,
                cond_logvar,
                self.gamma,
            )
        loss = mse + kld
        logs = {
            "Loss/val_loss": loss.item(),
            "Loss/val_mse": mse.item(),
            "Loss/val_kld": kld.item(),
        }
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch, full_val=False):
        # VAE eval: aggregate SSIM & LPIPS over full validation set
        device = next(self.parameters()).device
        first = epoch == 1
        if full_val:
            total_pixels = 0
            total_ssim = 0.0
            total_lpips = 0.0
            first_batch = True

            for batch in val_loader:
                y, x = batch
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    x_hat = self.sample(y, y.shape[0], gamma_added=True)
                b = x.size(0)
                total_pixels += b

                # per-sample metrics
                for orig, recon in zip(x, x_hat):
                    ssim = self.ssim(
                        orig.cpu().numpy(),
                        recon.cpu().numpy(),
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total_ssim += ssim
                    total_lpips += self.lpips_fn(
                        orig[[2, 1, 0]].unsqueeze(0), recon[[2, 1, 0]].unsqueeze(0)
                    ).item()

                # capture first batch for image logging
                if first_batch:
                    if first:
                        imgs_in = y[:4].clamp(0, 1)
                        bicubic = (
                            F.interpolate(y, scale_factor=2, mode="bicubic")[
                                :4, [2, 1, 0], :, :
                            ].clamp(0, 1),
                        )
                        imgs_out = x_hat[:4].clamp(0, 1)
                        imgs_gt = x[:4].clamp(0, 1)
                        wandb_run.log(
                            {
                                "Images/Bicubic": [
                                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                                    for img in bicubic[0]
                                ],
                                "Images/Input": [
                                    wandb.Image(
                                        img[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                    for img in imgs_in
                                ],
                                "Images/Original": [
                                    wandb.Image(
                                        img[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                    for img in imgs_gt
                                ],
                                "Images/Reconstruction": [
                                    wandb.Image(
                                        img[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                    for img in imgs_out
                                ],
                                "HyperParameters/Gamma": [
                                    wandb.Image(
                                        gamma[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .detach()
                                        .numpy()
                                    )
                                    for gamma in self.gamma[:4]
                                ]
                                if self.gamma_type != "scalar"
                                else [],
                            },
                            step=epoch,
                        )
                        first = False
                    else:
                        imgs_in = y[:4].clamp(0, 1)
                        imgs_out = x_hat[:4].clamp(0, 1)
                        wandb_run.log(
                            {
                                "Images/Reconstruction": [
                                    wandb.Image(
                                        img[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                    for img in imgs_out
                                ],
                                "HyperParameters/Gamma": [
                                    wandb.Image(
                                        gamma[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .detach()
                                        .numpy()
                                    )
                                    for gamma in self.gamma[:4]
                                ]
                                if self.gamma_type != "scalar"
                                else [],
                            },
                            step=epoch,
                        )
                    first_batch = False

            # compute averages
            avg_ssim = total_ssim / total_pixels
            avg_lpips = total_lpips / total_pixels

            # log aggregate metrics
            wandb_run.log(
                {"Metrics/SSIM": avg_ssim, "Metrics/LPIPS": avg_lpips},
                step=epoch,
            )

        else:
            batch = next(iter(val_loader))
            y, x = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x_hat = self.sample(y, y.shape[0], gamma_added=True)
                imgs_in = y[:4]
                imgs_out = x_hat[:4]

        # log sample images
        if epoch % 5 == 0:
            wandb_run.log(
                {
                    "Images/Reconstruction": [
                        wandb.Image(img[[2, 1, 0], :, :].permute(1, 2, 0).cpu().numpy())
                        for img in imgs_out
                    ],
                },
                step=epoch,
            )

    def on_train_epoch_end(self, **kwargs):
        self.wandb_run.log(
            {
                "HyperParameters/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )
        if self.gamma_type == "scalar":
            self.wandb_run.log(
                {
                    "HyperParameters/Gamma": self.gamma.item(),
                },
                step=self.current_epoch,
            )

    def on_train_start(self, **kwargs):
        if self.gamma_type == "scalar":
            self.gamma.requires_grad = True

        val_loader = self.val_loader
        device = next(self.parameters()).device

        if os.path.exists("baseline_ckpt.pth"):
            baseline = torch.load("baseline_ckpt.pth", weights_only=False)
            self.ssim_base = baseline["ssim_base"]
            self.lpips_base = baseline["lpips_base"]
            print(
                f"Baseline SSIM: {self.ssim_base}, LPIPS: {self.lpips_base}. Skipping baseline computation."
            )
            return
        else:
            ssim_cumu, lpips_cumu = 0, 0
            for _, batch in tqdm(enumerate(val_loader)):
                y_val, x_val = batch
                y_val, x_val = y_val.to(device), x_val.to(device)

                hr_interp = F.interpolate(y_val, scale_factor=2, mode="bicubic")

                # Compute SSIM and LPIPS scores
                for bcb, hr in zip(hr_interp, x_val):
                    ssim_val = self.ssim(
                        hr.cpu().numpy(),
                        bcb.cpu().numpy(),
                        data_range=1.0,
                        channel_axis=0,
                    )
                    lpips = self.lpips_fn(
                        hr[[2, 1, 0]].unsqueeze(0), bcb[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_cumu += ssim_val
                    lpips_cumu += lpips
            self.ssim_base = ssim_cumu / (len(val_loader.dataset))
            self.lpips_base = lpips_cumu / (len(val_loader.dataset))
            torch.save(
                {
                    "ssim_base": self.ssim_base,
                    "lpips_base": self.lpips_base,
                },
                "baseline_ckpt.pth",
            )
            print(
                f"Baseline SSIM: {self.ssim_base}, LPIPS: {self.lpips_base}. Baseline computation complete."
            )

    def get_task_data(self, val_loader):
        batch = next(iter(val_loader))
        y, x = batch
        x, y = (
            x.to(next(self.parameters()).device),
            y.to(next(self.parameters()).device),
        )
        x = x[2:3, :, :, :]  # Take the first sample from the batch
        y = y[2:3, :, :, :]
        return y, x

    def sample(self, y, samples=100, gamma_added=True, recurrent=True):
        """
        Generate samples from the VAE given a condition y.
        Args:
            y (torch.Tensor): Condition tensor of shape (batch_size, latent_size).
            samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, 4, patch_size, patch_size).
        """
        mu, logvar = self.cond_prior(y).chunk(2, dim=1)  # Split into mu and logvar
        mu, logvar = self.conv_condmu(mu), self.conv_condlogvar(logvar)
        _, _, h, w = y.shape
        z = torch.randn(
            samples, int(int(256 / (self.adjust * 2)) * 2 * h * w / 64), device=y.device
        )
        z = mu + torch.exp(0.5 * logvar) * z
        self.gamma = self.decoder_variance(z)
        if y.shape[0] == 1:
            y = y.expand(samples, -1, -1, -1)  # Expand x to match samples
        mean_decode = self.decode(z, y)
        """x_hat = (
            mean_decode + torch.randn_like(mean_decode) * self.gamma
            if gamma_added
            else mean_decode
        )
        """
        x_hat = laplacian_sampling(mean_decode, self.gamma.pow(2)) if gamma_added else mean_decode

        x_hat, *_ = self.forward(x_hat, y, self.L) if recurrent else x_hat
        x_hat = laplacian_sampling(x_hat, self.gamma.pow(2)) if gamma_added else x_hat
        return x_hat


if __name__ == "__main__":
    model = Cond_VAE(cr=1.5, patch_size=64)
    print(model)
    print(model.adjust)
    y = torch.randn(1, 4, 32, 32)
    x = torch.randn(1, 4, 64, 64)  # Example input tensor
    x_hat, mu, logvar, _ = model.forward(x, y)
    cond_mu, cond_logvar = model.cond_prior(y).chunk(
        2, dim=1
    )  # Split into mu and logvar
    print("Output shape:", x_hat.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("Cond Mu shape:", cond_mu.shape)
    print("Cond Logvar shape:", cond_logvar.shape)
