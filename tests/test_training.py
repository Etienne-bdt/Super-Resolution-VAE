import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import models.base as base_module
from models.mvae import Multimodal_VAE
from models.vae import VAE


class DummyWandbRun:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


@pytest.fixture(autouse=True)
def dummy_wandb(monkeypatch):
    # force all wandb.init calls to return a no-op run
    monkeypatch.setattr(
        base_module.wandb, "init", lambda *args, **kwargs: DummyWandbRun()
    )


def test_vae_training_loop_runs_one_epoch():
    cr = 2
    patch_size = 32
    x_data = torch.randn(2, 4, patch_size, patch_size)
    ds = TensorDataset(x_data, x_data)
    loader = DataLoader(ds, batch_size=2)
    model = VAE(cr=cr, patch_size=patch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # should finish without error
    model.fit(
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        optimizer=optimizer,
        epochs=1,
        start_epoch=1,
        val_metrics_every=1,
        slurm_job_id="test",
    )
    assert model.scheduler.last_epoch == 1


def test_cond_vae_training_loop_runs_one_epoch():
    cr = 2
    patch_size = 64
    # Initialize the Cond_SRVAE model with the calculated latent size
    x_data = torch.randn(2, 4, patch_size, patch_size)
    y_data = torch.randn(2, 4, patch_size // 2, patch_size // 2)
    ds = TensorDataset(y_data, x_data)
    loader = DataLoader(ds, batch_size=2)
    model = Multimodal_VAE(cr, patch_size=patch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.fit(
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        optimizer=optimizer,
        epochs=1,
        start_epoch=1,
        val_metrics_every=1,
        slurm_job_id="test",
    )
    assert model.scheduler.last_epoch == 1
