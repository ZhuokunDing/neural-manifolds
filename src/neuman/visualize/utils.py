"""Utility classes and functions for visualization."""

from matplotlib.animation import FFMpegWriter
from tqdm import tqdm


class TqdmWriter(FFMpegWriter):
    """FFMpegWriter with tqdm progress bar for video encoding."""

    def __init__(self, total_frames, *args, **kwargs):
        """
        Initialize TqdmWriter.

        Parameters
        ----------
        total_frames : int
            Total number of frames to encode.
        *args
            Additional positional arguments passed to FFMpegWriter.
        **kwargs
            Additional keyword arguments passed to FFMpegWriter.
        """
        self._progbar = None
        self.total_frames = total_frames
        super().__init__(*args, **kwargs)

    def setup(self, fig, outfile, dpi=None):
        """Set up the writer and initialize the progress bar."""
        self._progbar = tqdm(total=self.total_frames, desc="Saving video")
        return super().setup(fig, outfile, dpi=dpi)

    def grab_frame(self, **savefig_kwargs):
        """Grab a frame and update the progress bar."""
        self._progbar.update(1)
        return super().grab_frame(**savefig_kwargs)

    def finish(self):
        """Finish encoding and close the progress bar."""
        self._progbar.close()
        return super().finish()
