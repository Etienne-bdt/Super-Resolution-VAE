import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
    PerceiverConfig,
    PerceiverImagePreprocessor,
    PerceiverModel,
)

from dataset import init_dataloader


class PerceiverWrapper(nn.Module):
    def __init__(self):
        super(PerceiverWrapper, self).__init__()
        self.configuration = PerceiverConfig(image_size=256)
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
        return self.model(x).logits

if __name__ == "__main__":
    model = PerceiverWrapper()
    
    train_loader, val_loader = init_dataloader(
        "s2v"
    )
    epoch = 100
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    for epoch in range(epoch):
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
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")