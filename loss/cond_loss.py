import torch
import torch.nn.functional as F


def cond_loss(recon_y, y, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    n_y = recon_y.shape[1] * recon_y.shape[2] * recon_y.shape[3]
    mse = n_y * (
        F.mse_loss(recon_y, y, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1).mean()
    return mse, kld
