import torch

from loss import base_loss, multimodal_loss
from models import VAE, Multimodal_VAE


def test_vae_forward_and_loss_shapes():
    cr = 2
    patch_size = 16
    latent_size = int(4 * patch_size * patch_size // cr)
    # Initialize the VAE model with the calculated latent size
    model = VAE(cr, patch_size=patch_size)
    x = torch.randn(2, 4, patch_size, patch_size)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape == (2, latent_size)
    assert logvar.shape == (2, latent_size)
    mse, kld = base_loss(x_hat, x, mu, logvar, torch.tensor(1.0))
    assert isinstance(mse.item(), float)
    assert isinstance(kld.item(), float)


def test_cond_vae_forward_and_loss_shapes():
    cr = 2
    patch_size = 16
    latent_size = int(4 * patch_size * patch_size // cr)
    model = Multimodal_VAE(cr, patch_size=patch_size)
    x = torch.randn(2, 4, patch_size, patch_size)
    y = torch.randn(2, 4, patch_size // 2, patch_size // 2)
    x_hat, y_hat, mu_z, logvar_z, mu_u, logvar_u, mu_z_uy, logvar_z_uy = model(x, y)
    assert x_hat.shape == x.shape
    assert y_hat.shape == y.shape
    assert mu_u.shape == (2, latent_size // 4)
    assert mu_z.shape == (2, latent_size)
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
        model.gammax,
        model.gammay,
    )
    for val in (mse_x, kld_u, mse_y, kld_z):
        assert isinstance(val.item(), float)
