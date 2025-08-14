from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def normalize_image(image: torch.Tensor, quantile: float = 1) -> torch.Tensor:
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
        max_val = torch.quantile(
            image.flatten(1), quantile, dim=-1, keepdim=True
        ).unsqueeze(-1)
        normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    elif image.ndim == 4:
        min_val = image.amin(dim=(2, 3), keepdim=True)
        max_val = torch.quantile(
            image.flatten(2), quantile, dim=-1, keepdim=True
        ).unsqueeze(-1)
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


def get_contrast(
    image: torch.Tensor, quantile: float = 0.99
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the minimum and maximum pixel values of the image tensor.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        tuple: Minimum and maximum pixel values of the image.
    """
    if image.ndim == 3:
        min_val = image.amin(dim=(1, 2), keepdim=True)
        max_val = torch.quantile(
            image.flatten(1), quantile, dim=-1, keepdim=True
        ).unsqueeze(-1)
    elif image.ndim == 4:
        min_val = image.amin(dim=(2, 3), keepdim=True)
        max_val = torch.quantile(
            image.flatten(2), quantile, dim=-1, keepdim=True
        ).unsqueeze(-1)
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
    elif image.ndim == 3:
        img = image
    else:
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


def icp(samples: torch.Tensor, gt: torch.Tensor, quantiles: torch.Tensor):
    """
    Compute the empirical quantiles of the mean squared error per pixel.
    Args:
        samples (torch.Tensor): Samples from the model (N, C, H, W).
        gt (torch.Tensor): Ground truth image (C, H, W).
        quantiles (torch.Tensor): Quantiles to compute.
    Returns:
        Tuple: Empirical coverage and quantiles map.
    """
    b, c, h, w = samples.shape
    x_mmse = samples.mean(dim=0, keepdim=True)[0]  # (1, C, H, W)
    persample_mse = (
        F.mse_loss(x_mmse.unsqueeze(0).expand(b, -1, -1, -1), samples, reduction="none")
        .mean(1)
        .view(b, -1)
    )  # b, h*w
    distances, _ = torch.sort(persample_mse, dim=0)  # (N, npixels)

    # Vectorized computation of distance limits and empirical coverage
    distance_limits = torch.clamp((b * quantiles).long(), 0, b - 1)  # (num_quantiles,)
    distance_thresholds = distances[distance_limits]  # (num_quantiles, npixels)

    mse = F.mse_loss(x_mmse, gt[0], reduction="none").mean(0).view(h * w)  # (h*w)
    # Compute empirical coverage for all quantiles at once
    empirical_coverage = (mse.unsqueeze(0) < distance_thresholds).float().mean(dim=(1))
    empirical_coverage[-1] = 1.0

    return empirical_coverage


def quantile_map(samples: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    Compute the quantile map for the given samples and a given quantile.

    Args:
        samples (torch.Tensor): Samples from the model (N, C, H, W).
        quantile (float): Quantile to compute (between 0 and 1).

    Returns:
        torch.Tensor: Quantile map of shape (C, H, W).
    """
    b, c, h, w = samples.shape
    x_mmse = samples.mean(dim=0, keepdim=True)[0]  # (1, C, H, W)
    persample_mse = (
        F.mse_loss(x_mmse.unsqueeze(0).expand(b, -1, -1, -1), samples, reduction="none")
        .mean(1)
        .view(b, -1)
    ).sqrt()  # b, h*w
    distances, _ = torch.sort(persample_mse, dim=0)  # (N, npixels)

    # Vectorized computation of distance limits and empirical coverage
    distance_limit = int((b * quantile))  # (1)
    distance_thresholds = distances[distance_limit]  # (npixels)

    return distance_thresholds.view(h, w)  # (h, w)


def save_img(x: torch.Tensor, filename: str, false_color: bool = False) -> None:
    """
    Save the image tensor to a file.

    Args:
        x (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W).
        filename (str): Path to save the image.
        false_color (bool): If True, apply false color mapping.
    """
    if x.ndim == 4:
        x = x[0]
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError(
            "Input image must be a 2D (H, W), 3D (C, H, W) or 4D (B, C, H, W) tensor."
        )

    if not false_color and x.shape[0] == 4:
        # Apply false color mapping
        x = x[[2, 1, 0], :, :]
    else:
        x = x[[3, 2, 1], :, :]

    x_numpy = x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.imsave(filename, x_numpy)
    plt.close()

def laplacian_sampling(mu, b):
    """
    Sample from a Laplacian distribution with mean mu and scale b.

    Args:
        mu (torch.Tensor): Mean of the Laplacian distribution.
        b (torch.Tensor): Scale parameter of the Laplacian distribution.

    Returns:
        torch.Tensor: Samples from the Laplacian distribution.
    """
    u = torch.rand(size=mu.shape, device=mu.device) - 0.5
    return mu - b * torch.sign(u) * torch.log1p(-2 * u.abs())


def gaussian_sampling(mu, sigma):
    """
    Sample from a Gaussian distribution with mean mu and standard deviation sigma.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        sigma (torch.Tensor): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Samples from the Gaussian distribution.
    """
    eps = torch.randn_like(mu)
    return mu + sigma * eps