import argparse
import os
import time

import torch

import callbacks
import models
from dataset import init_dataloader


def main(args):
    """
    args : arguments from the command line
    args.epochs : number of epochs to train the model
    args.dataset : dataset to use for training
    """
    train_loader, val_loader = init_dataloader(
        args.dataset, args.batch_size, args.patch_size
    )
    cr = args.compression_ratio

    if cr <= 0:
        raise ValueError("Compression ratio must be a positive integer.")
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", f"local_{time.strftime('%Y%m%D-%H%M%S')}"
    )
    callbacks_list = [
        callbacks.ModelCheckpoint(
            slurm_job_id, "ckpt", monitor="Loss/val_loss", mode="min"
        ),
        callbacks.EarlyStopping(patience=25, delta=0.01),
    ]

    L = 1

    if args.model_type == "VAE":
        model = models.VAE(
            cr,
            args.patch_size // 2,
            callbacks=callbacks_list,
            slurm_job_id=slurm_job_id,
        )
    elif args.model_type == "MVAE":
        model = models.Multimodal_VAE(
            cr,
            args.patch_size,
            callbacks=callbacks_list,
            slurm_job_id=slurm_job_id,
            L=L,
        )
    elif args.model_type == "Cond_VAE":
        model = models.Cond_VAE(
            cr,
            args.patch_size,
            callbacks=callbacks_list,
            slurm_job_id=slurm_job_id,
            L=L,
            gamma_type=args.gamma_type,
        )

    else:
        raise ValueError(
            f"Unknown model type: {args.model_type}. Choose 'MVAE' or 'VAE'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.model_ckpt:
        print("Loading model from checkpoint...")
        save_dict = torch.load(args.model_ckpt)
        # start_epoch = save_dict["epoch"] + 1
        model.load_state_dict(save_dict)
        print("Model loaded successfully.")
        # print("Loading optimizer state...")
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        # print("Optimizer state loaded successfully.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 1

    if not (args.test and args.model_ckpt):
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            optimizer=optimizer,
            start_epoch=start_epoch,
            val_metrics_every=args.val_metrics_every,
            slurm_job_id=slurm_job_id,
            L=L,
        )

    model.task(val_loader)


def parse_args():
    """
    Parse command line arguments. Notably to set the number of epochs or change the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument(
        "--pre_epochs",
        type=int,
        default=20,
        help="Number of epochs to pre-train the low resolution model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--dataset", type=str, default="s2v", help="Type of the dataset"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Patch size of the High-Res Images.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, the model will be tested instead of trained.",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Path to the model checkpoint to resume training.",
    )

    parser.add_argument(
        "--val_metrics_every",
        type=int,
        default=5,
        help="Number of epochs between validation metrics computation.",
    )

    parser.add_argument(
        "-cr",
        "--compression_ratio",
        type=float,
        default=1.5,
        help="Compression of the ratio.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="MVAE",
        choices=["MVAE", "VAE", "Cond_VAE"],
        help="Model to use : 'MVAE' ou 'VAE' ou 'Cond_VAE'.",
    )

    parser.add_argument(
        "--gamma_type",
        type=str,
        default="scalar",
        choices=["scalar", "vector"],
        help="Type of gamma to use in the model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print("==========================")
    print("Initializing training with the following arguments:")
    print(arguments)
    print("--------------------------")
    print(
        f"Model checkpoint: {'not' if arguments.model_ckpt is None else arguments.model_ckpt} provided"
    )
    if arguments.model_ckpt:
        print("Checking if model exists...")
        if not os.path.exists(arguments.model_ckpt):
            raise FileNotFoundError(
                f"Model checkpoint {arguments.model_ckpt} not found."
            )
        else:
            print("Model checkpoint found.")
    print("--------------------------")
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================")
    main(args=arguments)
