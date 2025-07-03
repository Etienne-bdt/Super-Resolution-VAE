import torch
import torch.nn.functional as F


def multimodal_loss(
    recon_x, x, recon_y, y, mu1, logvar1, mu2, logvar2, mu3, logvar3, gammax, gammay
):
    """
    Loss function for the Conditional Super-Resolution VAE. This function minimizes the **NEGATIVE** ELBO:
    .. math::
        -(E[log p(x|z)] - D_KL(q(z|x)||p(z))) .

    Args:
        recon_x (torch.Tensor): Reconstructed high-resolution (HR) image.
        x (torch.Tensor): Original high-resolution (HR) image.
        recon_y (torch.Tensor): Reconstructed low-resolution (LR) image.
        y (torch.Tensor): Original low-resolution (LR) image.
        mu1 (torch.Tensor): Mean of the latent variable for the LR image.
        logvar1 (torch.Tensor): Log variance of the latent variable for the LR image.
        mu2 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the HR image.
        logvar2 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the HR image.
        mu3 (torch.Tensor): Mean of the latent variable for the HR image conditioned on the LR image.
        logvar3 (torch.Tensor): Log variance of the latent variable for the HR image conditioned on the LR image.
        gammax (torch.Tensor): Variance of the prior for the HR image.
        gammay (torch.Tensor): Variance of the prior for the LR image.

    Returns:
        tuple: A tuple containing:
            - mse_x (torch.Tensor): Mean squared error for the HR image.
            - kld_u (torch.Tensor): Kullback-Leibler divergence for the LR image.
            - mse_y (torch.Tensor): Mean squared error for the LR image.
            - kld_z (torch.Tensor): Kullback-Leibler divergence for the HR image.

    Note:
        Parameters 1 (mu1, logvar1) and 2,3 (mu2, logvar2, mu3, logvar3) represent the latent variables for `u` and `z`, respectively:
        - 2: `z` derived from `x`.
        - 3: `z` derived from `y` and `u`.
    """
    y_shape = recon_y.shape
    x_shape = recon_x.shape
    if recon_x.ndim == 5:
        x = x.expand(x_shape[0], -1, -1, -1, -1)
        y = y.expand(y_shape[0], -1, -1, -1, -1)
        n_y = y_shape[2] * y_shape[3] * y_shape[4]
        n_x = x_shape[2] * x_shape[3] * x_shape[4]
    else:
        n_y = y_shape[1] * y_shape[2] * y_shape[3]
        n_x = x_shape[1] * x_shape[2] * x_shape[3]
    mse_y = n_y * (
        F.mse_loss(recon_y, y, reduction="mean") / (2 * gammay.pow(2)) + (gammay.log())
    )
    kld_u = 0.5 * torch.sum(mu1.pow(2) + logvar1.exp() - 1 - logvar1, dim=1).mean()
    mse_x = n_x * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gammax.pow(2)) + (gammax.log())
    )
    kld_z = (
        0.5
        * (
            torch.sum(logvar3 - logvar2 - 1, dim=1)
            + torch.sum((logvar2 - logvar3).exp(), dim=1)
            + torch.sum((mu2 - mu3).pow(2) * ((-logvar3).exp()), dim=1)
        ).mean()
    )
    return mse_x, kld_u, mse_y, kld_z
