import torch


class BaseAdapter:
    def adapt(self, video: torch.Tensor) -> torch.Tensor:
        """
        Base method for adapting video inputs.

        Args:
            video (torch.Tensor): Input video tensor.

        Returns:
            torch.Tensor: Adapted video tensor.
        """
        raise NotImplementedError("Subclasses must implement the adapt method.")
