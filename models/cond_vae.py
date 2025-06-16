import os

import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from loss import cond_loss

from .base import BaseVAE
from .layers import down_block, up_block


class Cond_SRVAE(BaseVAE):
    def __init__(self, cr, patch_size=64, callbacks=None, slurm_job_id="local"):
        if callbacks is None:
            callbacks = []
        super(Cond_SRVAE, self).__init__(patch_size, callbacks, slurm_job_id)
        self.cr = cr
        self.latent_size = int((patch_size * patch_size * 4 / self.cr) // 256) * 256
        self.latent_size_y = self.latent_size // 4
        self.patch_size = patch_size
        self.gammax = torch.tensor(1.0, requires_grad=True)
        self.gammay = torch.tensor(1.0, requires_grad=True)

        self.encoder_y = nn.Sequential(
            down_block(in_channels=4, out_channels=16),  # out 16 , 16 , 16
            down_block(in_channels=16, out_channels=64),  # out 64, 8, 8
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=(self.latent_size_y // 64)
                * 2,  # Output channels for mu and logvar
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Flatten(start_dim=1),  # Flatten to (batch_size, latent_size // 8)
            # out 512 * 2 * 2 = 2048
        )

        self.decoder_y = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    self.latent_size_y // 64,
                    self.patch_size // 2**3,
                    self.patch_size // 2**3,
                ),
            ),
            up_block(
                in_channels=self.latent_size_y // 64,
                out_channels=128,
            ),
            up_block(
                in_channels=128,
                out_channels=64,
            ),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )

        self.encoder_x = nn.Sequential(
            down_block(in_channels=4, out_channels=16),
            down_block(in_channels=16, out_channels=64),
            down_block(
                in_channels=64,
                out_channels=128,
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=(self.latent_size // 64),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # out 1024 * 4 * 4 = 16384
        )  # out 1024, 4, 4

        self.z_carac = nn.Sequential(
            down_block(
                in_channels=self.latent_size * 3 // 64,
                out_channels=self.latent_size * 3 // 32,
            ),
            nn.Conv2d(
                in_channels=self.latent_size * 3 // 32,
                out_channels=self.latent_size * 3 // 32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=self.latent_size * 3 // 32,
                out_channels=self.latent_size * 3 // 16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=self.latent_size * 3 // 16,
                out_channels=self.latent_size * 2 // 16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=self.latent_size * 2 // 16,
                out_channels=self.latent_size * 2 // 16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Flatten(1),
        )

        self.decoder_x = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    self.latent_size * 3 // 64,
                    self.patch_size // 2**3,
                    self.patch_size // 2**3,
                ),
            ),
            up_block(
                in_channels=self.latent_size * 3 // 64,
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
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),  # Ensure output is in [0, 1]
        )

        self.y_to_z = nn.Sequential(
            down_block(in_channels=4, out_channels=16),
            down_block(in_channels=16, out_channels=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.latent_size // 64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # Flatten to (batch_size, latent_size // 3)
            # out 8192
        )
        # Replace Linear layers with Conv-based alternatives
        self.u_to_z = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    self.latent_size_y // 64,
                    self.patch_size // 2**3,
                    self.patch_size // 2**3,
                ),
            ),
            nn.Conv2d(
                self.latent_size_y // 64,
                self.latent_size_y // 64,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                self.latent_size_y // 64,
                self.latent_size_y // 32,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                self.latent_size_y // 32,
                self.latent_size_y // 32,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                self.latent_size_y // 32,
                self.latent_size_y // 16,
                kernel_size=3,
                padding=1,
            ),
        )

        self.uy_z = nn.Sequential(
            nn.Unflatten(
                1,
                (
                    self.latent_size * 2 // 16,
                    self.patch_size // 2**4,
                    self.patch_size // 2**4,
                ),
            ),
            nn.Conv2d(
                self.latent_size * 2 // 16,
                self.latent_size // 16,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                self.latent_size // 16, self.latent_size // 16, kernel_size=3, padding=1
            ),
            nn.Conv2d(
                self.latent_size // 16,
                self.latent_size * 2 // 16,
                kernel_size=3,
                padding=1,
            ),
            nn.Flatten(1),
        )

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Cond_SRVAE initialized with {self.num_params} trainable parameters.")

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
        stack = torch.cat((u, y, z), dim=1)
        return self.decoder_x(stack)

    def forward(self, x, y):
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)
        x_enc = self.encode_x(x)
        y_enc = self.y_to_z(y)
        u_enc = self.u_to_z(u)

        mu_z, logvar_z = torch.chunk(
            self.z_carac(torch.cat((x_enc, y_enc, u_enc), dim=1)), 2, dim=1
        )
        z = self.reparameterize(mu_z, logvar_z)

        y_enc = y_enc.view(y_enc.size(0), -1)
        u_enc = u_enc.view(u_enc.size(0), -1)
        mu_z_uy, logvar_z_uy = self.z_cond(y_enc, u_enc)

        x_hat = self.decode_x(z, y_enc, u_enc)
        y_hat = self.decode_y(u)

        return x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy

    def conditional_generation(self, y):
        # Generate a sample from the model
        mu_u, logvar_u = self.encode_y(y)
        u = self.reparameterize(mu_u, logvar_u)

        y = self.y_to_z(y).view(y.size(0), -1)
        u = self.u_to_z(u).view(u.size(0), -1)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)
        z = self.reparameterize(mu_z_uy, logvar_z_uy)

        x_hat = self.decode_x(z, y, u)
        return x_hat

    def sample(self, y, samples=1000) -> torch.Tensor:
        # Generate samples from the model
        mu_u, logvar_u = self.encode_y(y)

        std_u = torch.exp(0.5 * logvar_u)
        latent_u = std_u.size(1)
        eps_u = torch.randn((samples, latent_u)).to(y.device)

        u = mu_u + eps_u * std_u

        if y.ndim == 3:
            y = y.unsqueeze(0)
            # Using expand to not reallocate for the same tensor
            y = y.expand(u.size(0) - 1, -1, -1)
        elif y.ndim == 4 and y.size(0) == 1:
            y = y.expand(u.size(0), -1, -1, -1)
        y = self.y_to_z(y).view(y.size(0), -1)
        u = self.u_to_z(u).view(u.size(0), -1)

        mu_z_uy, logvar_z_uy = self.z_cond(y, u)

        mu_z_uy, logvar_z_uy = (
            mu_z_uy.mean(dim=0).unsqueeze(0),
            logvar_z_uy.mean(dim=0).unsqueeze(0),
        )

        std = torch.exp(0.5 * logvar_z_uy)
        latent = std.size(1)
        eps = torch.randn((samples, latent)).to(y.device)

        z = mu_z_uy + eps * std

        print(f"y shape: {y.shape}, u shape: {u.shape}, z shape: {z.shape}")
        return self.decode_x(z, y, u)

    def generation(self):
        u = torch.randn(1, self.latent_size_y).to("cuda")
        y = self.decode_y(u)

        return y, self.conditional_generation(y)

    def train_step(self, batch, device):
        y, x = batch
        y, x = y.to(device), x.to(device)
        x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = (
            self.forward(x, y)
        )
        mse_x, kld_u, mse_y, kld_z = cond_loss(
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
            mse_x, kld_u, mse_y, kld_z = cond_loss(
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
                    x_hat, y_hat, *_ = self.forward(x, y)
                    x_sr = self.conditional_generation(y)

                b = y.size(0)
                count += b

                for oy, ry, ox, rx, gen in zip(y, y_hat, x, x_hat, x_sr):
                    ssim_y = self.ssim(
                        oy.cpu().numpy(),
                        ry.cpu().numpy(),
                        win_size=11,
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
                        win_size=11,
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
                        win_size=11,
                        data_range=1.0,
                        channel_axis=0,
                    )
                    total["ssim_sr"] += ssim_sr
                    total["lpips_sr"] += self.lpips_fn(
                        ox[[2, 1, 0]].unsqueeze(0), gen[[2, 1, 0]].unsqueeze(0)
                    ).item()

                if first:
                    imgs = {
                        "y": y[:4],
                        "x": x[:4],
                        "y_bicubic": F.interpolate(y, scale_factor=2, mode="bicubic")[
                            :4
                        ],  # Bicubic interpolation for y
                        "y_hat": y_hat[:4],
                        "x_hat": x_hat[:4],
                        "x_sr": x_sr[:4],
                    }
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
                x_hat, y_hat, *_ = self.forward(x, y)
                x_sr = self.conditional_generation(y)

            imgs = {
                "y": y[:4],
                "x": x[:4],
                "y_bicubic": F.interpolate(y, scale_factor=2, mode="bicubic")[
                    :4
                ],  # Bicubic interpolation for y
                "y_hat": y_hat[:4],
                "x_hat": x_hat[:4],
                "x_sr": x_sr[:4],
            }

        if epoch % 10 == 0 or epoch == 1:
            # log sample images
            wandb_run.log(
                {
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
                },
                step=epoch,
            )

    def on_train_start(self, **kwargs):
        self.gammax.requires_grad = True
        self.gammay.requires_grad = True
        device = next(self.parameters()).device
        self.optimizer.add_param_group(
            {
                "params": [self.gammax, self.gammay],
            }
        )
        val_loader = self.val_loader
        if val_loader is None:
            raise ValueError(
                "Validation loader must be provided for baseline evaluation."
            )
        if os.path.exists("baseline_ckpt.pth"):
            baseline = torch.load("baseline_ckpt.pth")
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
                        hr.numpy(),
                        bcb.numpy(),
                        win_size=11,
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
        self.wandb_run.log(
            {
                "HyperParameters/Gamma_X": self.gammax.item(),
                "HyperParameters/Gamma_Y": self.gammay.item(),
                "HyperParameters/Learning Rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )

    def get_task_data(self, val_loader):
        batch = next(iter(val_loader))
        y, x = batch
        y, x = (
            y.to(next(self.parameters()).device),
            x.to(next(self.parameters()).device),
        )
        return y[7:8, :, :, :], x[
            7:8, :, :, :
        ]  # Return a single sample for task evaluation


if __name__ == "__main__":
    # Example usage
    model = Cond_SRVAE(cr=2, patch_size=64)
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

    # You can add more code here to test the model, e.g., training loop, etc.
    # Note: This is just a placeholder for demonstration purposes.
    pass
