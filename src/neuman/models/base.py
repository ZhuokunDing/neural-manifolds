from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pandas as pd


class BaseVideoModel(ABC, nn.Module):
    def __init__(self):
        """
        Abstract base class for video-to-neural activity models.

        Args:
            n_neurons (int): Number of neurons to predict
        """
        super().__init__()

    @abstractmethod
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            videos (torch.Tensor): Input videos of shape (batch_size, channels, time, height, width)

        Returns:
            torch.Tensor: Predicted neural activity of shape (batch_size, n_neurons, time)
        """
        pass

    @abstractmethod
    def neuron_info(self) -> pd.DataFrame:
        """
        Metadata about the neurons

        Returns:
            pd.DataFrame: Metadata about the neurons of shape (n_neurons, n_cols)
        """
        pass

    @abstractmethod
    def model_info(self) -> dict:
        """
        Metadata about the model

        Returns:
            dict: Metadata about the model
        """
        pass
