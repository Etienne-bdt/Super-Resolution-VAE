import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
    PerceiverConfig,
    PerceiverImagePreprocessor,
    PerceiverModel,
)
from skimage import metrics as skmetrics
from dataset import init_dataloader


class PerceiverWrapper(nn.Module):
    def __init__(self):
        super(PerceiverWrapper, self).__init__()
        self.configuration = PerceiverConfig(image_size=64)
        self.preprocessor = PerceiverImagePreprocessor(
            self.configuration,
            prep_type="conv1x1",
            spatial_downsample=1,
            in_channels=4,
            out_channels=256,
            position_encoding_type="trainable",
            concat_or_add_pos="concat",
            project_pos_dim=256,
            trainable_position_encoding_kwargs=dict(
                num_channels=256,
                index_dims=self.configuration.image_size**2,
            ),
        )
        self.decoder = PerceiverBasicDecoder(
            self.configuration,
            num_channels=self.configuration.d_latents,
            output_num_channels=4,
            output_index_dims=3,
            trainable_position_encoding_kwargs=dict(
                num_channels=self.configuration.d_latents,
                index_dims=(self.configuration.image_size*2)**2,
            ),
        )
        self.model = PerceiverModel(
            self.configuration,
            input_preprocessor=self.preprocessor,
            decoder=self.decoder,
        )

    def forward(self, x):
        return self.model(x).logits.view(-1, 4, self.configuration.image_size * 2, self.configuration.image_size * 2)

if __name__ == "__main__":
    model = PerceiverWrapper()
    
    train_loader, val_loader = init_dataloader(
        "s2v", batch_size=16, patch_size=128
    )
    ssim = skmetrics.structural_similarity
    epochs = 100
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wandb_run = wandb.init(project="Perceiver-SR", name=f"SLURM_{os.getenv('SLURM_JOB_ID')}",entity="ebardet-isae-supaero")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        wandb.log({"Training Loss": train_loss}, step=epoch)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            model.eval()
            val_loss = 0.0
            total_ssim = 0.0
            with torch.no_grad():
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                for orig,recon in zip(targets.cpu().numpy(), outputs.cpu().numpy()):
                    ssim_val = ssim(orig,recon,multichannel=True, channel_axis=0, data_range=1.0)
                    total_ssim += ssim_val
        ssim_val = total_ssim / len(val_loader.dataset)
        wandb.log({"SSIM": ssim_val}, step=epoch)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"Validation Loss": val_loss}, step=epoch)
        