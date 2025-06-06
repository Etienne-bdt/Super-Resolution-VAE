import abc
import os
from difflib import get_close_matches

import torch
import torch.nn as nn


class Callback(abc.ABC):
    """
    Abstract Class for defining callbacks in a training loop.
    """

    @abc.abstractmethod
    def on_epoch_begin(self, **kwargs):
        """
        Called at the beginning of each epoch.
        Returns: bool: True if training should stop, False otherwise.
        """
        return False

    @abc.abstractmethod
    def on_epoch_end(self, **kwargs):
        """
        Called at the end of each epoch.
        Returns: bool: True if training should stop, False otherwise.
        """
        return False


class EarlyStopping(Callback):
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
        self.metric_name = "val_loss"

    def on_epoch_begin(self, **kwargs) -> bool:
        """
        Call this method at the beginning of each epoch.
        Returns: bool: True if training should stop, False otherwise.
        """
        return False

    def on_epoch_end(self, **kwargs) -> bool:
        """
        Call this method to check if training should be stopped.
        Args:
            val_loss (float): Current validation loss.
        """
        logs = kwargs.get("logs", {})
        val_loss = logs.get(self.metric_name, float("inf"))
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ModelCheckpoint(Callback):
    """
    Class to save the model at the end of each epoch.
    """

    def __init__(
        self,
        slurm_job_id: str,
        save_path: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        """
        Args:
            save_path (str): Path to save the model.
            monitor (str): Metric to monitor for saving the model.
            mode (str): One of {'min', 'max'}. In 'min' mode, the model is saved when the monitored metric decreases.
            save_best_only (bool): If True, only saves the model when the monitored metric improves.
        """
        self.slurm_job_id = slurm_job_id
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def on_epoch_begin(self, **kwargs) -> bool:
        return False

    def on_epoch_end(self, **kwargs) -> bool:
        """
        Called at the end of each epoch to save the model if the monitored metric improves.
        Args:
            logs (dict): Dictionary containing the metrics for the epoch.
        """
        logs = kwargs.get("logs", {})
        epoch = kwargs.get("epoch", 0)
        model = kwargs.get("model", nn.Sequential())
        if epoch == 1:
            # find the closest match to monitor in logs
            if self.monitor not in logs:
                close_matches = get_close_matches(
                    self.monitor, logs.keys(), n=1, cutoff=0
                )[0]
                if close_matches:
                    self.monitor = close_matches
                else:
                    raise ValueError(
                        f"Monitor metric '{self.monitor}' not found in logs. Available metrics: {list(logs.keys())}"
                    )

        current_metric = logs.get(self.monitor, float("inf"))
        if self.save_best_only:
            if (self.mode == "min" and current_metric < self.best_metric) or (
                self.mode == "max" and current_metric > self.best_metric
            ):
                self.best_metric = current_metric
                self.best_epoch = kwargs.get("epoch", 0)
                # Save the model here
                torch.save(
                    model.state_dict(),
                    os.path.join(self.save_path, f"{self.slurm_job_id}.pth"),
                )
        else:
            # Save the model every epoch
            torch.save(
                model.state_dict(),
                os.path.join(
                    self.save_path,
                    f"{self.slurm_job_id}_epoch_{kwargs.get('epoch', 0)}.pth",
                ),
            )
        return False
