import torch
import torch.nn.functional as F


def cond_loss(recon_x, x, mu, logvar, cond_mu, cond_logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    n_x = recon_x.shape[1] * recon_x.shape[2] * recon_x.shape[3]
    mse = n_x * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld = (
        0.5
        * (
            torch.sum(cond_logvar - logvar - 1, dim=1)
            + torch.sum((logvar - cond_logvar).exp(), dim=1)
            + torch.sum((mu - cond_mu).pow(2) * ((-cond_logvar).exp()), dim=1)
        ).mean()
    )
    return mse, kld
