import os

import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from loss import multimodal_loss

from .base import BaseVAE
from .layers import conv_block, down_block, up_block


class Multimodal_VAE(BaseVAE):
    def __init__(
        self,
        cr,
        patch_size=64,
        callbacks=None,
        slurm_job_id="local",
        L=100,
        gamma_type="scalar",
    ):
        if callbacks is None:
            callbacks = []
        super(Multimodal_VAE, self).__init__(patch_size, callbacks, slurm_job_id)
        self.cr = cr
        self.adjustx = self.cr
        self.adjusty = 4 * self.cr
        self.patch_size = patch_size
        self.L = L
        self.gamma_type = gamma_type
        self.distribution_type = "gaussian"
        if gamma_type == "scalar":
            self.gammax = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.gammay = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            # Add variance decoder for x (HR output)
            self.variance_decoder_x = nn.Sequential(
                nn.Unflatten(
                    1,
                    (
                        int(256 / (2 * self.adjustx)) * 2,
                        patch_size // 2**4,
                        patch_size // 2**4,
                    ),
                ),
                up_block(
                    in_channels=int(256 / (2 * self.adjustx)) * 2,
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
            # Add variance decoder for y (LR output)
            self.variance_decoder_y = nn.Sequential(
                nn.Unflatten(
                    1,
                    (
                        int(256 / (2 * self.adjusty)) * 2,
                        patch_size // 2**4,
                        patch_size // 2**4,
                    ),
                ),
                up_block(
                    in_channels=int(256 / (2 * self.adjusty)) * 2,
                    out_channels=256,
                ),
                up_block(
                    in_channels=256,
                    out_channels=128,
                ),
                up_block(128, 64),
                conv_block(64, 32, 1, 1, 0),
                conv_block(32, 16, 1, 1, 0),
                conv_block(
                    16, 4, 1, 1, 0, final_relu=False
                ),  # Final conv to match input channels
                nn.Sigmoid(),  # Ensure output is in [0, 1]
            )

        self.encoder_y = nn.Sequential(
            down_block(in_channels=4, out_channels=16),  # out 16,
            down_block(in_channels=16, out_channels=64),  # out 64, 16, 16
            down_block(64, 128),
            conv_block(128, 256, 3, 1, 1),
            conv_block(
                256,
                int(256 / (2 * self.adjusty)) * 4,
                1,
                1,
                0,
                final_relu=False,
            ),
            nn.Flatten(start_dim=1),  # Flatten to (batch_size, latent_size // 8)
            # out 512 * 2 * 2 = 2048
        )

        self.decoder_y = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    int(256 / (2 * self.adjusty)) * 2,
                    patch_size // 2**4,
                    patch_size // 2**4,
                ),
            ),
            up_block(
                in_channels=int(256 / (2 * self.adjusty)) * 2,
                out_channels=256,
            ),
            up_block(
                in_channels=256,
                out_channels=128,
            ),
            up_block(128, 64),
            conv_block(64, 32, 1, 1, 0),
            conv_block(32, 16, 1, 1, 0),
            conv_block(
                16, 4, 1, 1, 0, final_relu=False
            ),  # Final conv to match input channels
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )

        self.hr_down = down_block(in_channels=4, out_channels=16)  # out 16 , 16 , 16

        self.encoder_x = nn.Sequential(
            down_block(in_channels=20, out_channels=64),  # out 64, 8, 8
            down_block(64, 128),
            down_block(128, 256),
            conv_block(
                256,
                int(256 / (2 * self.adjustx)) * 2,
                1,
                1,
                0,
                final_relu=False,
            ),
            # Flatten to (batch_size, latent_size // 8)
            # out 512 * 2 * 2 = 2048
        )

        self.decoder_x = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    int(256 / (2 * self.adjustx)) * 4,
                    patch_size // 2**4,
                    patch_size // 2**4,
                ),
            ),
            up_block(
                in_channels=int(256 / (2 * self.adjustx)) * 4,
                out_channels=512,
            ),
            up_block(
                in_channels=512,
                out_channels=256,
            ),
            up_block(256, 128),
            conv_block(128, 64, 1, 1, 0),
            conv_block(64, 32, 1, 1, 0),
        )
        self.decoder_end = nn.Sequential(
            up_block(in_channels=36, out_channels=16),  # upsample to 8x8
            conv_block(16, 8, 3, 1, 1),
            conv_block(8, 8, 1, 1, 0),
            conv_block(
                8, 4, 1, 1, 0, final_relu=False
            ),  # Final conv to match input channels
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )

        self.z_carac = nn.Sequential(
            down_block(
                in_channels=int(256 / (2 * self.adjustx)) * 4,
                out_channels=int(256 / (2 * self.adjustx)) * 16,
            ),
            conv_block(
                in_channels=int(256 / (2 * self.adjustx)) * 16,
                out_channels=int(256 / (2 * self.adjustx)) * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                final_relu=False,
            ),
            nn.Flatten(1),
        )

        self.y_to_z = nn.Sequential(
            down_block(in_channels=4, out_channels=16),
            down_block(in_channels=16, out_channels=64),
            down_block(64, 128),
            conv_block(128, int(256 / (2 * self.adjustx)) * 2, 3, 1, 1),
            # Flatten to (batch_size, latent_size // 3)
            # out 8192
        )
        # Replace Linear layers with Conv-based alternatives
        self.u_to_z = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    int(256 / (2 * self.adjusty)) * 2,
                    self.patch_size // 2**4,
                    self.patch_size // 2**4,
                ),
            ),
            conv_block(
                in_channels=int(256 / (2 * self.adjusty)) * 2,
                out_channels=int(256 / (2 * self.adjusty)) * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            conv_block(
                in_channels=int(256 / (2 * self.adjusty)) * 4,
                out_channels=int(256 / (2 * self.adjustx)) * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                final_relu=False,
            ),  # Flatten to (batch_size, latent_size // 3)
        )

        self.uy_z = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    int(256 / (2 * self.adjustx)) * 4,
                    self.patch_size // 2**4,
                    self.patch_size // 2**4,
                ),
            ),
            conv_block(
                in_channels=int(256 / (2 * self.adjustx)) * 4,
                out_channels=int(256 / (2 * self.adjusty)) * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            conv_block(
                in_channels=int(256 / (2 * self.adjusty)) * 4,
                out_channels=int(256 / (2 * self.adjustx)) * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                final_relu=False,
            ),
            nn.Flatten(1),
        )

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(
            f"Multimodal VAE initialized with {self.num_params} trainable parameters."
        )

    def z_cond(self, y, u):
        # Define the encoder part of the VAE

        jointure = torch.cat((y, u), dim=1)

        uy_z = self.uy_z(jointure)

        return torch.chunk(uy_z, 2, dim=1)

    def encode_y(self, y):
        # Define the encoder part of the VAE
        y = self.encoder_y(y)
        return torch.chunk(y, 2, dim=1)

    def encode_x(self, x):
        # Define the encoder part of the VAE
        return self.encoder_x(x)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_y(self, u):
        return self.decoder_y(u)

    def decode_x(self, z, y, u):
        stack = torch.cat((u, z), dim=1)
        x_decode = self.decoder_x(stack)
        end_stack = torch.cat((x_decode, y), dim=1)
        return self.decoder_end(end_stack)

    def decoder_variance_x(self, z):
        if self.gamma_type == "scalar":
            return self.gammax
        else:
            return self.variance_decoder_x(z)

    def decoder_variance_y(self, u):
        if self.gamma_type == "scalar":
            return self.gammay
        else:
            return self.variance_decoder_y(u)

    def forward(self, x, y, L=None):
        if L is None:
            L = self.L
        mu_u, logvar_u = self.encode_y(y)
        x_hat_list = []
        y_hat_list = []

        x_pre = self.hr_down(x)
        # Encode x and y
        x_y_stack = torch.cat((x_pre, y), dim=1)

        x_enc = self.encode_x(x_y_stack)
        y_enc = self.y_to_z(y)
        # Reparameterization trick for u
        # This loop is added to allow multiple reparameterizations of u
        # This is useful for sampling from the latent space
        u = self.reparameterize(mu_u, logvar_u)
        u_enc = self.u_to_z(u)
        stack = torch.cat((x_enc, u_enc), dim=1)
        mu_z, logvar_z = torch.chunk(self.z_carac(stack), 2, dim=1)
        y_enc = y_enc.view(y_enc.size(0), -1)
        u_enc = u_enc.view(u_enc.size(0), -1)
        mu_z_uy, logvar_z_uy = self.z_cond(y_enc, u_enc)
        for _ in range(L):
            z = self.reparameterize(mu_z, logvar_z)
            x_hat = self.decode_x(z, y, u_enc)
            y_hat = self.decode_y(u)
            x_hat_list.append(x_hat.unsqueeze(0))
            y_hat_list.append(y_hat.unsqueeze(0))
        if L > 1:
            x_hat_5d = torch.cat(x_hat_list, dim=0)
            y_hat_5d = torch.cat(y_hat_list, dim=0)

        else:
            x_hat_5d = x_hat
            y_hat_5d = y_hat

        # Get variances
        gammax = self.decoder_variance_x(z)
        gammay = self.decoder_variance_y(u)
        self.gammax = gammax
        self.gammay = gammay

        return x_hat_5d, y_hat_5d, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy

    def conditional_generation(self, y):
        # Generate a sample from the model
        y_original = y
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        y = self.y_to_z(y).view(y.size(0), -1)
        u = self.u_to_z(u).view(u.size(0), -1)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)
        z = self.reparameterize(mu_z_uy, logvar_z_uy)

        x_hat = self.decode_x(z, y_original, u)
        return x_hat

    def sample(self, y, samples=100, gamma_added=True, recurrent=False):
        # Generate a sample from the model
        y_original = y
        B, c, h, w = y.size()
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        y = self.y_to_z(y).view(y.size(0), -1)
        u = self.u_to_z(u).view(u.size(0), -1)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)

        latent_dim = mu_z_uy.size(1)
        device = y.device
        z = torch.randn(samples, B, latent_dim, device=device)
        z = mu_z_uy.unsqueeze(0) + z * torch.exp(0.5 * logvar_z_uy.unsqueeze(0))

        z_flat = z.reshape(samples * B, latent_dim)
        u_rep = u.expand(samples, -1, -1).reshape(samples * B, -1)

        x_hat = self.decode_x(
            z_flat,
            y_original.expand(samples, -1, -1, -1, -1).reshape(samples * B, c, h, w),
            u_rep,
        )
        x_hat = x_hat.view(samples, B, *x_hat.shape[1:])

        if gamma_added:
            gx = self.decoder_variance_x(z_flat)
            if self.gamma_type == "scalar":
                self.gammax = gx
            else:
                gammax = gx.view(samples, B, 4, h * 2, w * 2)
                self.gammax = gammax
            self.gamma = self.gammax
            return self.sample_from_distribution(
                x_hat, self.gammax, gamma_added=gamma_added
            )
        else:
            return x_hat

    def train_step(self, batch, device):
        y, x = batch
        y, x = y.to(device), x.to(device)
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = (
            self.forward(x, y)
        )
        mse_x, kld_u, mse_y, kld_z = multimodal_loss(
            x_hat,
            x,
            y_hat,
            y,
            mu_u,
            logvar_u,
            mu_z,
            logvar_z,
            mu_z_uy,
            logvar_z_uy,
            self.gammax,
            self.gammay,
        )
        loss = mse_x + kld_u + mse_y + kld_z
        logs = {
            "Loss/loss": loss.item(),
            "Loss/mse_x": mse_x.item(),
            "Loss/kld_u": kld_u.item(),
            "Loss/mse_y": mse_y.item(),
            "Loss/kld_z": kld_z.item(),
        }
        return loss, logs

    def val_step(self, batch, device):
        y, x = batch
        y, x = y.to(device), x.to(device)
        with torch.no_grad():
            x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = self(
                x, y
            )
            mse_x, kld_u, mse_y, kld_z = multimodal_loss(
                x_hat,
                x,
                y_hat,
                y,
                mu_u,
                logvar_u,
                mu_z,
                logvar_z,
                mu_z_uy,
                logvar_z_uy,
                self.gammax,
                self.gammay,
            )
            loss = mse_x + kld_u + mse_y + kld_z
        logs = {
            "Loss/val_loss": loss.item(),
            "Loss/val_mse_x": mse_x.item(),
            "Loss/val_kld_u": kld_u.item(),
            "Loss/val_mse_y": mse_y.item(),
            "Loss/val_kld_z": kld_z.item(),
        }
        return loss, logs

    def evaluate(self, val_loader, wandb_run, epoch, full_val=False):
        # CondVAE eval: aggregate SSIM & LPIPS over full validation set

        device = next(self.parameters()).device

        if full_val:
            total = {
                "ssim_y": 0.0,
                "lpips_y": 0.0,
                "ssim_x": 0.0,
                "lpips_x": 0.0,
                "ssim_sr": 0.0,
                "lpips_sr": 0.0,
            }
            count = 0
            first = True
            for batch in val_loader:
                y, x = [t.to(device) for t in batch]
                with torch.no_grad():
                    x_hat, y_hat, *_ = self.forward(x, y, L=1)
                    # Use sample method for conditional generation with consistency
                    x_sr_samples = self.sample(y, 100, gamma_added=False)
                    if x_sr_samples.ndim == 5:
                        x_sr = x_sr_samples.mean(dim=0)  # Average over samples
                    else:
                        x_sr = x_sr

                b = y.size(0)
                count += b

                for oy, ry, ox, rx, gen in zip(y, y_hat, x, x_hat, x_sr):
                    ssim_y = self.ssim(
                        oy.cpu().numpy(),
                        ry.cpu().numpy(),
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_y"] += ssim_y
                    total["lpips_y"] += self.lpips_fn(
                        oy[[2, 1, 0]].unsqueeze(0), ry[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_x = self.ssim(
                        ox.cpu().numpy(),
                        rx.cpu().numpy(),
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_x"] += ssim_x
                    total["lpips_x"] += self.lpips_fn(
                        ox[[2, 1, 0]].unsqueeze(0), rx[[2, 1, 0]].unsqueeze(0)
                    ).item()
                    ssim_sr = self.ssim(
                        ox.cpu().numpy(),
                        gen.cpu().numpy(),
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_sr"] += ssim_sr
                    total["lpips_sr"] += self.lpips_fn(
                        ox[[2, 1, 0]].unsqueeze(0), gen[[2, 1, 0]].unsqueeze(0)
                    ).item()

                if first:
                    imgs = {
                        "y": y[:4, [2, 1, 0], :, :].clamp(0, 1),
                        "x": x[:4, [2, 1, 0], :, :].clamp(0, 1),
                        "y_bicubic": F.interpolate(y, scale_factor=2, mode="bicubic")[
                            :4, [2, 1, 0], :, :
                        ].clamp(0, 1),  # Bicubic interpolation for y
                        "y_hat": y_hat[:4, [2, 1, 0], :, :].clamp(0, 1),
                        "x_hat": x_hat[:4, [2, 1, 0], :, :].clamp(0, 1),
                        "x_sr": x_sr[:4, [2, 1, 0], :, :].clamp(0, 1),
                    }
                    log_dict = {
                        "Images/LR_Input": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["y"]
                        ],
                        "Images/HR_Input": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["x"]
                        ],
                        "Images/LR_Bicubic": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["y_bicubic"]
                        ],
                        "Images/LR_Recon": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["y_hat"]
                        ],
                        "Images/HR_Recon": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["x_hat"]
                        ],
                        "Images/SR_Output": [
                            wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                            for img in imgs["x_sr"]
                        ],
                    }

                    # Add gamma images if not scalar (matching cond_vae pattern)
                    if self.gamma_type != "scalar":
                        log_dict.update(
                            {
                                "HyperParameters/Gamma_X": [
                                    wandb.Image(
                                        gamma[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .detach()
                                        .numpy()
                                    )
                                    for gamma in self.gammax[:4]
                                ],
                                "HyperParameters/Gamma_Y": [
                                    wandb.Image(
                                        gamma[[2, 1, 0], :, :]
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .detach()
                                        .numpy()
                                    )
                                    for gamma in self.gammay[:4]
                                ],
                            }
                        )

                    wandb_run.log(log_dict, step=epoch)
                    first = False

            # average metrics
            avg = {k: total[k] / count for k in total}

            # log aggregate metrics
            wandb_run.log(
                {
                    "Metrics/SSIM_LR": avg["ssim_y"],
                    "Metrics/LPIPS_LR": avg["lpips_y"],
                    "Metrics/SSIM_HR": avg["ssim_x"],
                    "Metrics/LPIPS_HR": avg["lpips_x"],
                    "Metrics/SSIM_SR": avg["ssim_sr"],
                    "Metrics/LPIPS_SR": avg["lpips_sr"],
                    "Metrics/SSIM_Baseline": self.ssim_base,
                    "Metrics/LPIPS_Baseline": self.lpips_base,
                },
                step=epoch,
            )
        else:
            batch = next(iter(val_loader))
            y, x = [t.to(device) for t in batch]
            with torch.no_grad():
                x_hat, y_hat, *_ = self.forward(x, y, L=1)
                x_sr_samples = self.sample(y, 100, gamma_added=False)
                if x_sr_samples.ndim == 5:
                    x_sr = x_sr_samples.mean(dim=0)
                else:
                    x_sr = x_sr_samples

            imgs = {
                "y_hat": y_hat[:4, [2, 1, 0], :, :].clamp(0, 1),
                "x_hat": x_hat[:4, [2, 1, 0], :, :].clamp(0, 1),
                "x_sr": x_sr[:4, [2, 1, 0], :, :].clamp(0, 1),
            }

        if epoch % 5 == 0:  # Changed from 10 to 5 to match cond_vae
            # log sample images
            log_dict = {
                "Images/LR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["y_hat"]
                ],
                "Images/HR_Recon": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_hat"]
                ],
                "Images/SR_Output": [
                    wandb.Image(img.permute(1, 2, 0).cpu().numpy())
                    for img in imgs["x_sr"]
                ],
            }

            # Add gamma images if not scalar (matching cond_vae pattern)
            if self.gamma_type != "scalar":
                log_dict.update(
                    {
                        "HyperParameters/Gamma_X": [
                            wandb.Image(
                                gamma[[2, 1, 0], :, :]
                                .permute(1, 2, 0)
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            for gamma in self.gammax[:4]
                        ],
                        "HyperParameters/Gamma_Y": [
                            wandb.Image(
                                gamma[[2, 1, 0], :, :]
                                .permute(1, 2, 0)
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            for gamma in self.gammay[:4]
                        ],
                    }
                )

            wandb_run.log(log_dict, step=epoch)

    def on_train_start(self, **kwargs):
        if self.gamma_type == "scalar":
            self.gammax.requires_grad = True
            self.gammay.requires_grad = True

        device = next(self.parameters()).device
        val_loader = self.val_loader
        if val_loader is None:
            raise ValueError(
                "Validation loader must be provided for baseline evaluation."
            )
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
            self.ssim_base = ssim_cumu / x_val.shape[0] * len(val_loader)
            self.lpips_base = lpips_cumu / x_val.shape[0] * len(val_loader)
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

    def on_train_epoch_end(self, **kwargs):
        log_dict = {
            "HyperParameters/Learning Rate": self.scheduler.get_last_lr()[0],
        }

        if self.gamma_type == "scalar":
            log_dict.update(
                {
                    "HyperParameters/Gamma_X": self.gammax.item(),
                    "HyperParameters/Gamma_Y": self.gammay.item(),
                }
            )

        self.wandb_run.log(log_dict, step=self.current_epoch)

    def get_task_data(self, val_loader):
        batch = next(iter(val_loader))
        y, x = batch
        y, x = (
            y.to(next(self.parameters()).device),
            x.to(next(self.parameters()).device),
        )
        return y[2:3, :, :, :], x[
            2:3, :, :, :
        ]  # Return a single sample for task evaluation


if __name__ == "__main__":
    # Example usage
    model = Multimodal_VAE(cr=1.5, patch_size=64, gamma_type="vector")
    print(model)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 4, 32, 32)  # Example condition

    x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = model(x, y)
    print("x_hat shape:", x_hat.shape)
    print("y_hat shape:", y_hat.shape)
    print("mu_z shape:", mu_z.shape)
    print("logvar_z shape:", logvar_z.shape)
    print("mu_u shape:", mu_u.shape)
    print("logvar_u shape:", logvar_u.shape)
    print("mu_z_uy shape:", mu_z_uy.shape)
    print("logvar_z_uy shape:", logvar_z_uy.shape)

    if model.gamma_type != "scalar":
        gammax = model.gammax
        gammay = model.gammay
        print("Gamma X shape:", gammax.shape)
        print("Gamma Y shape:", gammay.shape)

    # You can add more code here to test the model, e.g., training loop, etc.
    # Note: This is just a placeholder for demonstration purposes.
    pass
