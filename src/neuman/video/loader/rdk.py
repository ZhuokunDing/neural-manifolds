import numpy as np
import torch
import pandas as pd
from typing import Tuple, Optional
from .base import BaseLoader
from neuman.video import Video


class RDKLoader(BaseLoader):
    """
    Random Dot Kinematogram loader that generates rotating dot patterns.

    Creates the illusion of rotation through temporal coherence of square
    white dots on black background. Dots are positioned uniformly randomly
    across the frame, making each static frame appear as white noise while
    maintaining temporal coherence for the rotation illusion.

    Parameters
    ----------
    width : int, optional
        Frame width in pixels (default: 64)
    height : int, optional
        Frame height in pixels (default: 36)
    duration : float, optional
        Video duration in seconds (default: 10.0)
    fps : float, optional
        Frames per second (default: 30.0)
    num_dots : int, optional
        Number of dots in the RDK (default: 100)
    coherence : float, optional
        Fraction of dots moving coherently, range [0, 1] (default: 1.0)
    rotation_speed : float, optional
        Rotation speed in degrees per second (default: 36.0)
    dot_radius : int, optional
        DEPRECATED: Dots are now fixed 2x2 pixel squares. This parameter
        is kept for backward compatibility but is no longer used. (default: 2)
    disk_radius : float or None, optional
        Maximum radius of the rotating disk as a fraction of the minimum
        frame dimension. If None, dots can appear anywhere in the frame.
        If a float (e.g., 0.4), dot centers are constrained to a circular
        region with radius = disk_radius * min(width, height). Note that
        2x2 dot squares may extend slightly beyond this radius. (default: None)
    seed : int, optional
        Random seed for reproducibility (default: None)

    Notes
    -----
    Dots are rendered as 2x2 pixel squares and are positioned uniformly
    randomly across the entire frame. Coherent dots rotate around the frame
    center, creating the rotation illusion through temporal coherence.

    Examples
    --------
    >>> # Create loader with default parameters (dots across entire frame)
    >>> loader = RDKLoader()
    >>> video = loader.load_video()

    >>> # Create loader with custom parameters
    >>> loader = RDKLoader(width=128, height=128, coherence=0.8, seed=42)
    >>> video = loader.load_video()

    >>> # Create loader with constrained disk radius
    >>> loader = RDKLoader(disk_radius=0.3, num_dots=100, seed=42)
    >>> video = loader.load_video()  # Dots constrained to 30% of min dimension
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 36,
        duration: float = 10.0,
        fps: float = 30.0,
        num_dots: int = 100,
        coherence: float = 1.0,
        rotation_speed: float = 36.0,
        dot_radius: int = 2,
        disk_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        # Validate parameters
        if not 0.0 <= coherence <= 1.0:
            raise ValueError(f"Coherence must be in [0, 1], got {coherence}")
        if num_dots <= 0:
            raise ValueError(f"num_dots must be positive, got {num_dots}")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if disk_radius is not None and disk_radius <= 0:
            raise ValueError(f"disk_radius must be positive or None, got {disk_radius}")

        # Store parameters
        self.width = width
        self.height = height
        self.duration = duration
        self.fps = float(fps)
        self.num_dots = num_dots
        self.coherence = coherence
        self.rotation_speed = rotation_speed
        self.dot_radius = dot_radius
        self.disk_radius = disk_radius
        self.seed = seed

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate video frames and metadata
        self._frames, self._metadata = self._generate_rdk()

        # Convert to torch tensor (uint8)
        self._frames = torch.tensor(self._frames, dtype=torch.uint8)

        # Create Video object and cache it
        self._video = Video(
            frames=self._frames,
            metadata=self._metadata,
            fps=self.fps
        )

    def _initialize_dots(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize coherent dot parameters with infinite lifetime.

        Calculates uniform density and distributes dots between inside/outside
        disk regions. Stores dot counts as instance variables for rendering.

        Returns
        -------
        dot_radii : np.ndarray of shape (num_coherent,)
            Distance from center for each coherent dot (in pixels)
        dot_angles_0 : np.ndarray of shape (num_coherent,)
            Initial angular position for each coherent dot (in radians)
        """
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        frame_area = self.width * self.height

        if self.disk_radius is None:
            # Original behavior: all dots across frame
            num_coherent = int(self.coherence * self.num_dots)
            self.num_coherent = num_coherent
            self.num_random_inside = 0
            self.num_random_outside = self.num_dots - num_coherent

            # Initialize coherent dots uniformly across frame
            if num_coherent > 0:
                dot_x_0 = np.random.uniform(0, self.width, num_coherent)
                dot_y_0 = np.random.uniform(0, self.height, num_coherent)

                dx = dot_x_0 - center_x
                dy = dot_y_0 - center_y
                dot_radii = np.sqrt(dx**2 + dy**2)
                dot_angles_0 = np.arctan2(dy, dx)
            else:
                dot_radii = np.array([])
                dot_angles_0 = np.array([])

        else:
            # Calculate uniform density
            density = self.num_dots / frame_area

            # Calculate disk area
            max_radius_px = self.disk_radius * min(self.width, self.height)
            disk_area = np.pi * max_radius_px**2
            # Clip disk area to frame area (in case disk_radius >= 1.0)
            disk_area = min(disk_area, frame_area)
            outside_area = frame_area - disk_area

            # Calculate dots per region (maintaining uniform density)
            num_in_disk = int(density * disk_area)
            num_outside_disk = int(density * outside_area)

            # Within disk: split between coherent and random
            num_coherent = int(self.coherence * num_in_disk)
            num_random_inside = num_in_disk - num_coherent

            # Store counts for rendering
            self.num_coherent = num_coherent
            self.num_random_inside = num_random_inside
            self.num_random_outside = num_outside_disk

            # Initialize coherent dots within disk using rejection sampling
            if num_coherent > 0:
                dot_x_0 = []
                dot_y_0 = []

                while len(dot_x_0) < num_coherent:
                    # Generate candidate positions
                    n_needed = num_coherent - len(dot_x_0)
                    # Oversample to reduce iterations
                    n_candidates = int(n_needed * 2 + 10)

                    x_candidates = np.random.uniform(0, self.width, n_candidates)
                    y_candidates = np.random.uniform(0, self.height, n_candidates)

                    # Calculate distances from center
                    dx_candidates = x_candidates - center_x
                    dy_candidates = y_candidates - center_y
                    dist_candidates = np.sqrt(dx_candidates**2 + dy_candidates**2)

                    # Keep only those within disk radius
                    valid_mask = dist_candidates <= max_radius_px
                    dot_x_0.extend(x_candidates[valid_mask])
                    dot_y_0.extend(y_candidates[valid_mask])

                # Convert to arrays and trim to exact number needed
                dot_x_0 = np.array(dot_x_0[:num_coherent])
                dot_y_0 = np.array(dot_y_0[:num_coherent])

                # Convert to polar coordinates
                dx = dot_x_0 - center_x
                dy = dot_y_0 - center_y
                dot_radii = np.sqrt(dx**2 + dy**2)
                dot_angles_0 = np.arctan2(dy, dx)
            else:
                dot_radii = np.array([])
                dot_angles_0 = np.array([])

        return dot_radii, dot_angles_0

    def _render_frame(
        self,
        frame_idx: int,
        dot_radii: np.ndarray,
        dot_angles_0: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Render a single frame of the RDK.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based)
        dot_radii : np.ndarray
            Distance from center for each coherent dot
        dot_angles_0 : np.ndarray
            Initial angular position for each coherent dot

        Returns
        -------
        frame : np.ndarray of shape (H, W), dtype=uint8
            Rendered frame with dots
        metadata_dict : dict
            Metadata for this frame
        """
        # Create black canvas
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        # Calculate current rotation angle
        time_sec = frame_idx / self.fps
        angle_deg = self.rotation_speed * time_sec
        angle_rad = np.deg2rad(angle_deg)

        # Frame center
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        # Render coherent dots: rotate around center
        if self.num_coherent > 0:
            coherent_angles = dot_angles_0 + angle_rad

            coherent_x = center_x + dot_radii * np.cos(coherent_angles)
            coherent_y = center_y + dot_radii * np.sin(coherent_angles)

            # Clip to frame boundaries
            coherent_x = np.clip(coherent_x, 0, self.width - 1)
            coherent_y = np.clip(coherent_y, 0, self.height - 1)

            # Convert to integer coordinates
            coherent_x = coherent_x.astype(int)
            coherent_y = coherent_y.astype(int)

            # Render coherent dots
            for x, y in zip(coherent_x, coherent_y):
                self._draw_dot(frame, x, y)

        # Generate random dots INSIDE disk (if disk_radius is set)
        if self.num_random_inside > 0 and self.disk_radius is not None:
            max_radius_px = self.disk_radius * min(self.width, self.height)
            random_inside_x = []
            random_inside_y = []

            while len(random_inside_x) < self.num_random_inside:
                n_needed = self.num_random_inside - len(random_inside_x)
                n_candidates = int(n_needed * 2 + 10)

                x_candidates = np.random.uniform(0, self.width, n_candidates)
                y_candidates = np.random.uniform(0, self.height, n_candidates)

                dx_candidates = x_candidates - center_x
                dy_candidates = y_candidates - center_y
                dist_candidates = np.sqrt(dx_candidates**2 + dy_candidates**2)

                valid_mask = dist_candidates <= max_radius_px
                random_inside_x.extend(x_candidates[valid_mask])
                random_inside_y.extend(y_candidates[valid_mask])

            random_inside_x = np.array(random_inside_x[:self.num_random_inside])
            random_inside_y = np.array(random_inside_y[:self.num_random_inside])

            random_inside_x = np.clip(random_inside_x, 0, self.width - 1).astype(int)
            random_inside_y = np.clip(random_inside_y, 0, self.height - 1).astype(int)

            for x, y in zip(random_inside_x, random_inside_y):
                self._draw_dot(frame, x, y)

        # Generate random dots OUTSIDE disk (if disk_radius is set)
        if self.num_random_outside > 0:
            if self.disk_radius is not None:
                max_radius_px = self.disk_radius * min(self.width, self.height)
                random_outside_x = []
                random_outside_y = []

                while len(random_outside_x) < self.num_random_outside:
                    n_needed = self.num_random_outside - len(random_outside_x)
                    n_candidates = int(n_needed * 2 + 10)

                    x_candidates = np.random.uniform(0, self.width, n_candidates)
                    y_candidates = np.random.uniform(0, self.height, n_candidates)

                    dx_candidates = x_candidates - center_x
                    dy_candidates = y_candidates - center_y
                    dist_candidates = np.sqrt(dx_candidates**2 + dy_candidates**2)

                    valid_mask = dist_candidates > max_radius_px
                    random_outside_x.extend(x_candidates[valid_mask])
                    random_outside_y.extend(y_candidates[valid_mask])

                random_outside_x = np.array(random_outside_x[:self.num_random_outside])
                random_outside_y = np.array(random_outside_y[:self.num_random_outside])

                random_outside_x = np.clip(random_outside_x, 0, self.width - 1).astype(int)
                random_outside_y = np.clip(random_outside_y, 0, self.height - 1).astype(int)

                for x, y in zip(random_outside_x, random_outside_y):
                    self._draw_dot(frame, x, y)
            else:
                # No disk constraint - random dots anywhere
                random_x = np.random.randint(0, self.width, self.num_random_outside)
                random_y = np.random.randint(0, self.height, self.num_random_outside)

                for x, y in zip(random_x, random_y):
                    self._draw_dot(frame, x, y)

        # Create metadata dict for this frame
        metadata_dict = {
            'frame_idx': frame_idx,
            'time_sec': time_sec,
            'rotation_angle_deg': angle_deg,
            'rotation_angle_rad': angle_rad,
            'coherence': self.coherence,
            'num_dots': self.num_dots,
            'num_coherent': self.num_coherent,
            'num_random_inside': self.num_random_inside,
            'num_random_outside': self.num_random_outside,
        }

        return frame, metadata_dict

    def _draw_dot(self, frame: np.ndarray, x: int, y: int) -> None:
        """
        Draw a 2x2 square dot on the frame at position (x, y).

        Parameters
        ----------
        frame : np.ndarray
            Frame array to draw on (modified in-place)
        x : int
            X coordinate of dot center
        y : int
            Y coordinate of dot center
        """
        # Calculate 2x2 square bounds (centered on x, y)
        x_min = max(0, x - 1)
        x_max = min(self.width, x + 1)
        y_min = max(0, y - 1)
        y_max = min(self.height, y + 1)

        # Set 2x2 block to white (255)
        frame[y_min:y_max+1, x_min:x_max+1] = 255

    def _generate_rdk(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate RDK frames and metadata.

        Returns
        -------
        frames : np.ndarray of shape (1, T, H, W), dtype=uint8
            Grayscale video frames
        metadata : pd.DataFrame
            Per-frame metadata with one row per frame
        """
        # Calculate number of frames
        num_frames = int(self.duration * self.fps)

        # Initialize dot parameters (infinite lifetime)
        dot_radii, dot_angles_0 = self._initialize_dots()

        # Pre-allocate frames array
        frames = np.zeros((num_frames, self.height, self.width), dtype=np.uint8)

        # List to collect metadata dicts
        metadata_list = []

        # Generate each frame
        for frame_idx in range(num_frames):
            frame, metadata_dict = self._render_frame(
                frame_idx, dot_radii, dot_angles_0
            )
            frames[frame_idx] = frame
            metadata_list.append(metadata_dict)

        # Convert to (C, T, H, W) format: add channel dimension
        frames = frames[np.newaxis, :, :, :]  # Shape: (1, T, H, W)

        # Create metadata DataFrame
        metadata = pd.DataFrame(metadata_list)

        return frames, metadata

    def load_video(self) -> Video:
        """
        Load the cached RDK video.

        Returns
        -------
        Video
            The RDK video object with frames, metadata, and fps
        """
        return self._video
