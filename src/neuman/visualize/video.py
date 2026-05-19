import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import HTML
import torch


def play_torch_video(video_tensor, fps=30):
    """
    Play a 4D PyTorch tensor as video using Matplotlib.

    Parameters:
        video_tensor (torch.Tensor): A 4D PyTorch tensor with shape (C, T, H, W).
        fps (int): Frames per second for the video playback.
    """
    if video_tensor.ndim != 4:
        raise ValueError(
            "Input video_tensor must be a 4D PyTorch tensor with shape (C, T, H, W)."
        )

    channels, num_frames, height, width = video_tensor.shape

    if channels != 1 and channels != 3:
        raise ValueError(
            "Input video_tensor must have 1 or 3 channels (grayscale or RGB)."
        )

    # Lock vmin/vmax based on the dtype's natural range so that a constant
    # first frame (e.g. a blank/grey lead-in) doesn't degenerate the norm and
    # clip later frames.
    if video_tensor.dtype == torch.uint8:
        vmin, vmax = 0, 255
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots()
    ax.axis("off")  # Turn off axes for better visualization

    if channels == 1:
        img = ax.imshow(
            video_tensor[0, 0].cpu().numpy(),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
    else:  # channels == 3
        img = ax.imshow(
            video_tensor[:, 0].permute(1, 2, 0).cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

    def update(frame):
        if channels == 1:
            img.set_array(video_tensor[0, frame].cpu().numpy())
        else:  # channels == 3
            img.set_array(video_tensor[:, frame].permute(1, 2, 0).cpu().numpy())
        return (img,)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=True)
    return HTML(ani.to_jshtml())


# Example usage:
if __name__ == "__main__":
    # Create a dummy video array with random values
    num_frames, channels, height, width = 100, 3, 64, 64

    # Create a dummy video tensor with random values
    video_tensor = torch.rand(channels, num_frames, height, width)

    # Play the video
    play_torch_video(video_tensor, fps=30)
