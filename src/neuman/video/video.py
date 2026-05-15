import torch
import pandas as pd


class Video:
    """
    Video class for handling video data.
    """

    def __init__(self, frames: torch.Tensor, metadata: pd.DataFrame, fps: float = 30.0):
        """
        Parameters
        ----------
        frames : torch.Tensor of size (C, T, H, W)
            Frames in the video.
        metadata : pd.DataFrame
            Metadata associated with the video frames.
        fps : int, optional
            Frames per second for the video, default is 30.
        """
        self.frames = frames
        self.fps = float(fps)
        self.duration = frames.shape[1] / fps  # duration in seconds
        self.metadata = metadata

    def play(self):
        """
        Play the video using a visualization library.
        This method can be overridden by subclasses to provide specific playback functionality.
        """
        from neuman.visualize.video import play_torch_video

        return play_torch_video(self.frames, fps=self.fps)

    def save(self, filepath, bitrate=1000, codec="h264", dpi=100):
        """
        Save the video as an MP4 file.

        Parameters
        ----------
        filepath : str or Path
            Path where the video will be saved (should end in .mp4).
        bitrate : int, optional
            Video bitrate in kbps, default is 1000.
        codec : str, optional
            Video codec to use, default is "h264". Other options include "mpeg4", "libx264".
        dpi : int, optional
            Dots per inch for the saved video, default is 100.
        """
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        from neuman.visualize.utils import TqdmWriter

        channels, num_frames, height, width = self.frames.shape

        if channels != 1 and channels != 3:
            raise ValueError(
                "Video must have 1 or 3 channels (grayscale or RGB)."
            )

        # Create figure with appropriate size based on video dimensions
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Initialize the image
        if channels == 1:
            img = ax.imshow(
                self.frames[0, 0].cpu().numpy(), cmap="gray", interpolation="nearest"
            )
        else:  # channels == 3
            img = ax.imshow(
                self.frames[:, 0].permute(1, 2, 0).cpu().numpy(), interpolation="nearest"
            )

        def update(frame):
            if channels == 1:
                img.set_array(self.frames[0, frame].cpu().numpy())
            else:  # channels == 3
                img.set_array(self.frames[:, frame].permute(1, 2, 0).cpu().numpy())
            return (img,)

        ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 / self.fps, blit=True)

        writer = TqdmWriter(
            fps=self.fps,
            bitrate=bitrate,
            total_frames=num_frames,
            codec=codec
        )
        ani.save(filepath, writer=writer, dpi=dpi)
        plt.close(fig)

        print(f"Video saved to {filepath}")
