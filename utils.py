from typing import Tuple

import torch


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image tensor to the range [0, 1] for visualization.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    if image.ndim == 3:
        # If the image is 3D (C, H, W)
        min_val = image.amin(dim=(1, 2), keepdim=True)
        max_val = image.amax(dim=(1, 2), keepdim=True)
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    elif image.ndim == 4:
        min_val = image.amin(dim=(2, 3), keepdim=True)
        max_val = image.amax(dim=(2, 3), keepdim=True)
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    else:
        raise ValueError("Input image must be 3D or 4D tensor.")
    return normalized_image


def norm_and_copy_image_dynamics(source: torch.Tensor, target: torch.Tensor):
    """
    Normalize the source image and copy it to the target tensor.
    Args:
        source (torch.Tensor): Source image tensor.
        target (torch.Tensor): Target image tensor to copy normalized values into.
    Returns:
        torch.Tensor: Target tensor with normalized values from the source.
    """
    if source.ndim == 3:
        # If the source is 3D (C, H, W)
        min_val = source.amin(dim=(1, 2), keepdim=True)
        max_val = source.amax(dim=(1, 2), keepdim=True)
        normalized_source = (source - min_val) / (max_val - min_val + 1e-5)
        target = (target - target.mean(dim=(1, 2), keepdim=True)) * (
            normalized_source.std(dim=(1, 2), keepdim=True)
            / target.std(dim=(1, 2), keepdim=True)
        ) + normalized_source.mean(dim=(1, 2), keepdim=True)

    elif source.ndim == 4:
        # If the source is 4D (B, C, H, W)
        min_val = source.amin(dim=(2, 3), keepdim=True)
        max_val = source.amax(dim=(2, 3), keepdim=True)
        normalized_source = (source - min_val) / (max_val - min_val + 1e-5)
        target = (target - target.mean(dim=(2, 3), keepdim=True)) * (
            normalized_source.std(dim=(2, 3), keepdim=True)
            / target.std(dim=(2, 3), keepdim=True)
        ) + normalized_source.mean(dim=(2, 3), keepdim=True)

    return normalized_source, target


def get_contrast(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the minimum and maximum pixel values of the image tensor.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        tuple: Minimum and maximum pixel values of the image.
    """
    if image.ndim == 3:
        min_val = image.amin(dim=(1, 2), keepdim=True)
        max_val = image.amax(dim=(1, 2), keepdim=True)
    elif image.ndim == 4:
        min_val = image.amin(dim=(2, 3), keepdim=True)
        max_val = image.amax(dim=(2, 3), keepdim=True)
    else:
        raise ValueError("Input image must be 3D or 4D tensor.")
    return (min_val, max_val)


def save_img_histogram(image: torch.Tensor, filename: str) -> None:
    """
    Save the histogram of pixel values of the image tensor.
    Args:
        image (torch.Tensor): Input image tensor.
        filename (str): Path to save the histogram image.
    """
    import matplotlib.pyplot as plt

    if image.ndim == 4:
        img = image[0]  # Take the first image in the batch
    elif image.ndim == 2:
        img = image.unsqueeze(0)
    elif image.ndim != 3:
        raise ValueError(
            "Input image must be a 2D (H, W), 3D (C, H, W) or 4D (B, C, H, W) tensor."
        )

    for i in range(img.shape[0]):
        plt.hist(
            img[i].flatten().cpu().numpy(),
            bins=256,
            range=(0, 1),
            alpha=0.5,
            label=f"Channel {i}",
        )

    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


class EarlyStopper:
    """
    Early stopping utility to stop training when validation loss does not improve.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, delta: float = 0) -> None:
        """
        Early stopping utility to stop training when validation loss does not improve.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0

    def __call__(self, val_loss: float) -> bool:
        """
        Call this method to check if training should be stopped.
        Args:
            val_loss (float): Current validation loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
