import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile
import torch
from torch.utils.data import Dataset

from utils import get_contrast, normalize_image


def init_dataloader(dataset: str, batch_size: int = 16, patch_size: int = 256):
    """
    Initialize the dataloader for the dataset.
    Args:
        dataset (str): The name of the dataset to load.
        batch_size (int): The batch size for the dataloader.
    Returns:
        train_loader (DataLoader): The dataloader for the training set.
        val_loader (DataLoader): The dataloader for the validation set.
    """
    if dataset == "Sen2Venus" or dataset == "sen2venus" or dataset == "s2v":
        ds = Sen2VenDataset(patch_size)

    elif dataset == "Floods" or dataset == "floods":
        ds = FloodDataset(patch_size=256)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    train_size = int(0.8 * len(ds))
    train_ds = torch.utils.data.Subset(ds, range(train_size))
    val_ds = torch.utils.data.Subset(ds, range(train_size, len(ds)))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=6,
        persistent_workers=True,
    )
    return train_loader, val_loader


class FloodDataset(Dataset):
    def __init__(self, patch_size=64):
        super(FloodDataset, self).__init__()
        self.patch_size = patch_size
        self.patches = []
        self.precompute_patches()

    def precompute_patches(self):
        root = os.listdir("/scratch/disc/e.bardet/Simple-VAE-RS/floods")
        for path in root:
            img_paths = [
                os.path.join(
                    "/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2", x
                )
                for x in os.listdir(
                    os.path.join(
                        "/scratch/disc/e.bardet/Simple-VAE-RS/floods", path, "S2"
                    )
                )
                if x.endswith(".tif")
            ]
            for img_path in img_paths:
                img = tifffile.imread(img_path)  # Read all bands
                height, width = img.shape[1], img.shape[2]
                for row in range(0, height, self.patch_size):
                    for col in range(0, width, self.patch_size):
                        if (
                            row + self.patch_size <= height
                            and col + self.patch_size <= width
                        ):
                            patch = img[
                                :,
                                row : row + self.patch_size,
                                col : col + self.patch_size,
                            ]
                            quantiles = np.quantile(
                                patch, [0.01, 0.99], axis=(1, 2), keepdims=True
                            )
                            patch = (patch - quantiles[0]) / (
                                quantiles[1] - quantiles[0] + 1e-5
                            )
                            patch = np.clip(patch, 0, 1)
                            patch = torch.tensor(patch, dtype=torch.float32)
                            if not torch.isnan(patch).any():
                                self.patches.append(patch)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


class Sen2VenDataset(Dataset):
    def __init__(self, patch_size=256, crop="random", dataset="ARM", bands="visu"):
        super(Sen2VenDataset, self).__init__()
        self.dataset = os.path.join(os.getcwd(), dataset)
        csv_path = os.path.join(self.dataset, "index.csv")
        self.df = pl.read_csv(csv_path, has_header=True, separator="	")
        self.patch_size = patch_size
        self.crop = crop
        if crop not in ["grid", "random"]:
            raise ValueError("Crop must be 'grid' or 'random'")
        if bands == "visu":
            self.df = self.df.select(["b2b3b4b8_10m", "b2b3b4b8_05m"])
            self.p0 = "b2b3b4b8_10m"
            self.p1 = "b2b3b4b8_05m"
        else:
            raise NotImplementedError(
                "Only 'visu' bands are implemented. Please choose 'visu'."
            )

        if crop == "grid":
            self.prepare_patches()

        assert patch_size <= 256, "Patch size must be less than or equal to 256"
        if patch_size < 256 and patch_size > 0 and patch_size % 2 == 0:
            # TODO: implement random cropping
            self.transform = True
        elif patch_size == 256:
            self.transform = False
        else:
            raise ValueError("Patch size must be a positive even number")

    def __len__(self):
        if self.crop == "grid":
            return len(self.patches)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if self.crop == "grid":
            item_path = self.patches[idx][0]
            patch_idx = self.patches[idx][1]
            p1 = item_path[self.p0].to_numpy()[0]
            p2 = item_path[self.p1].to_numpy()[0]

            p1 = os.path.join(self.dataset, p1)
            p2 = os.path.join(self.dataset, p2)

            # Load the images using rasterio
            img1 = tifffile.imread(p1)
            img2 = tifffile.imread(p2)

            img1 = torch.tensor(img1, dtype=torch.float32)
            img2 = torch.tensor(img2, dtype=torch.float32)

            (img1, img2) = (
                normalize_image(
                    self.select_crop(img1, self.patch_size // 2, patch_idx)
                ),
                normalize_image(self.select_crop(img2, self.patch_size, patch_idx)),
            )
            return img1, img2

        else:
            item_path = self.df[idx]
            p1 = item_path[self.p0].to_numpy()[0]
            p2 = item_path[self.p1].to_numpy()[0]

            p1 = os.path.join(self.dataset, p1)
            p2 = os.path.join(self.dataset, p2)

            # Load the images using rasterio
            img1 = tifffile.imread(p1, ioworkers=6)
            img2 = tifffile.imread(p2, ioworkers=6)

            img1 = torch.tensor(img1, dtype=torch.float32)
            img2 = torch.tensor(img2, dtype=torch.float32)

            if self.transform:
                if self.crop == "random":
                    img1, img2 = self.sr_randomcrop(img1, img2)
                elif self.crop == "grid":
                    img1 = self.grid_crop(img1, self.patch_size // 2)
                    img2 = self.grid_crop(img2, self.patch_size)

            min_img, max_img = get_contrast(img2)

            img2 = (img2 - min_img) / (max_img - min_img)
            img1 = ((img1 - min_img) / (max_img - min_img)).clamp(0, 1)

            return img1, img2

    def sr_randomcrop(self, img1, img2):
        """
        Randomly crop the images to the specified patch size. Images will share the same portion of the image, the first image will be cropped to half the patch size.
        The second image will be cropped to the full patch size.
        Args:
            img1 (torch.Tensor): The first image tensor.
            img2 (torch.Tensor): The second image tensor.
        Returns:
            img1 (torch.Tensor): The cropped first image tensor.
            img2 (torch.Tensor): The cropped second image tensor.
        """

        # Randomly crop the images to the specified patch size
        _, h, w = img1.shape
        top = np.random.randint(0, h - self.patch_size // 2)
        left = np.random.randint(0, w - self.patch_size // 2)
        img1 = img1[
            :, top : top + self.patch_size // 2, left : left + self.patch_size // 2
        ]
        img2 = img2[
            :,
            2 * top : 2 * top + self.patch_size,
            2 * left : 2 * left + self.patch_size,
        ]

        return img1, img2

    def select_crop(self, img, patch_size, index):
        num_patches = img.shape[2] // patch_size
        # With square images, the number of patches is the same in both dimensions
        row, col = index // num_patches, index % num_patches
        return img[
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ]

    def grid_crop(self, img, patch_size):
        """
        Crop the image into a grid of patches of the specified patch size.

        Args:
            img (torch.Tensor): The image tensor to crop.
            patch_size (int): The size of each patch.
        Returns:
            List[torch.Tensor]: A list of cropped patches.
        """
        _, h, w = img.shape
        patches = []
        for row in range(0, h, patch_size):
            for col in range(0, w, patch_size):
                if row + patch_size <= h and col + patch_size <= w:
                    patch = img[:, row : row + patch_size, col : col + patch_size]
                    patches.append(patch)
        return torch.stack(patches, dim=0)

    def prepare_patches(self):
        df = self.df
        item_path = df[0]
        p2 = item_path[self.p1].to_numpy()[0]

        p2 = os.path.join(self.dataset, p2)

        img2 = tifffile.imread(p2)
        img2_x = img2.shape[1]
        num_patches = (img2_x // self.patch_size) ** 2

        self.patches = []
        for i in range(len(df)):
            self.patches.extend((df[i], j) for j in range(num_patches))


def grid_collate(batch):
    """
    Custom collate function to handle the grid of patches.
    Args:
        batch (list): A list of batches.
    Returns:
        torch.Tensor: A tensor containing the concatenated patches.
    """
    t1, t2 = zip(*batch)
    return (torch.cat(t1, dim=0), torch.cat(t2, dim=0))


if __name__ == "__main__":
    # Example usage
    ds = Sen2VenDataset(patch_size=64)
    print(f"Number of samples: {len(ds)}")
    start = 15
    end = start + 1
    for i in range(start, end):
        img1, img2 = ds[i]
        if img1.ndim == 4:
            # If grid cropping is used, the images will have 4 dimensions instead of 3
            # in this case, we take the first image of the batch for the test
            img1, img2 = img1[0], img2[0]
        print(f"Image 1 shape: {img1.shape}, Image 2 shape: {img2.shape}")
        plt.imsave(
            f"img1_{i}.png",
            (img1[[2, 1, 0], :, :]).permute(1, 2, 0).numpy(),
        )
        plt.imsave(
            f"img2_{i}.png",
            (img2[[2, 1, 0], :, :]).permute(1, 2, 0).numpy(),
        )

    plt.figure()
    plt.hist(img1[0].numpy().flatten(), bins=100, alpha=0.5, label="b2", color="blue")
    plt.hist(img1[1].numpy().flatten(), bins=100, alpha=0.5, label="b3", color="green")
    plt.hist(img1[2].numpy().flatten(), bins=100, alpha=0.5, label="b4", color="red")
    plt.hist(img1[3].numpy().flatten(), bins=100, alpha=0.5, label="b8", color="orange")
    plt.title("Histogram of bands in Sentinel-2")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("histogram_sen2.png")

    plt.figure()
    plt.hist(img2[0].numpy().flatten(), bins=100, alpha=0.5, label="b3", color="blue")
    plt.hist(img2[1].numpy().flatten(), bins=100, alpha=0.5, label="b4", color="green")
    plt.hist(img2[2].numpy().flatten(), bins=100, alpha=0.5, label="b7", color="red")
    plt.hist(
        img2[3].numpy().flatten(), bins=100, alpha=0.5, label="b11", color="orange"
    )
    plt.title("Histogram of bands in Venus")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("histogram_venus.png")

    train_loader, val_loader = init_dataloader(
        "Sen2Venus", batch_size=16, patch_size=64
    )
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
