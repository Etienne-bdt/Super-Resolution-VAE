import torch
import torch.nn.functional as F


def cond_loss(recon_x, x, mu, logvar, cond_mu, cond_logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    x_shape = recon_x.shape
    if recon_x.ndim == 5:
        x = x.expand(x_shape[0], -1, -1, -1, -1)

    # Compute MSE per pixel
    mse_per_pixel = F.mse_loss(recon_x, x, reduction="none").to(
        gamma.device
    )  # (L, batch, chan, h, w)

    if isinstance(gamma, torch.nn.Parameter):
        gamma_reshaped = gamma
    else:
        # Reshape gamma to match the spatial dimensions of x
        gamma_reshaped = (
            gamma.view(x_shape[1], x_shape[2], x_shape[3], x_shape[4]).to(gamma.device)
            if recon_x.ndim == 5
            else gamma.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).to(
                gamma.device
            )
        )  # (1, chan, h, w)

    # Apply diagonal gamma weighting
    if recon_x.ndim == 5:
        mse = torch.sum(
            mse_per_pixel / (2 * gamma_reshaped.pow(2)) + gamma_reshaped.log(),
            dim=(2, 3, 4),  # Sum over spatial dimensions
        ).mean()  # Average over batch and L dimensions
    else:
        mse = torch.sum(
            mse_per_pixel / (2 * gamma_reshaped.pow(2)) + gamma_reshaped.log(),
            dim=(1, 2, 3),  # Sum over spatial dimensions
        ).mean()

    kld = (
        0.5
        * (
            torch.sum(cond_logvar - logvar - torch.ones_like(logvar), dim=1)
            + torch.sum((logvar - cond_logvar).exp(), dim=1)
            + torch.sum((mu - cond_mu).pow(2) * ((-cond_logvar).exp()), dim=1)
        ).mean()
    )
    return mse, kld
