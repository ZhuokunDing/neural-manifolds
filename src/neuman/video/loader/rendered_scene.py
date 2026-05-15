import numpy as np
import torch
from .base import BaseLoader
from neuman.video import Video
from typing import List
import pandas as pd
import datajoint as dj


class RenderedScenesLoader(BaseLoader):
    def __init__(self, dataset_hash: str, fps: int = 30):
        self.image_schema = dj.create_virtual_module(
            "pipeline-rendered-images", "pipeline_rendered_images"
        )
        dj.config.setdefault("stores", {})["rendered"] = dict(
            protocol="file",
            location="/mnt/dj-stor01/pipeline-externals",
        )
        self.dataset_hash = dataset_hash
        self._frames, self._metadata = self.load_images(dataset_hash)
        self._frames = torch.tensor(self._frames, dtype=torch.uint8)
        self._video = Video(self._frames, fps=fps, metadata=self._metadata)

    def load_images(self, dataset_hash: str):
        scenes, metadata = (
            self.image_schema.RenderedScenes & f"dataset_hash = '{dataset_hash}'"
        ).fetch("scene", "metadata")
        idx = np.argsort([m["img_counter"] for m in metadata])
        scenes, metadata = (
            scenes[idx],
            [metadata[i] for i in idx],
        )  # scenes are [H, W, C]
        scenes = np.stack(scenes, axis=0).transpose(
            3, 0, 1, 2
        )  # Convert to (C, T, H, W) format

        metadata = pd.DataFrame(metadata)
        metadata = metadata.assign(
            frame_idx=lambda df: df["img_counter"],
            rotation_degree=lambda df: df["img_counter"] / df["img_counter"].max() * 360,
        ) # HACK: rotation_degree is hardcoded here for convenience
        return scenes, metadata

    def load_video(self) -> Video:
        """
        Load the video from the dataset.

        Returns
        -------
        Video
            The loaded video object.
        """
        return self._video
