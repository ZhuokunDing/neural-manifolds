from .base import BaseAdapter
from dataclasses import dataclass
from torchvision import transforms
from neuman.video import Video
import torch


@dataclass
class FoundationModelInfo:
    period: float
    offset: float
    height: int
    width: int
    resize_id: str


class FoundationAdapter(BaseAdapter):
    def __init__(self, model_info: dict, aspect_ratio_tolerance: float = 0.001):
        super().__init__()
        self.model_info = FoundationModelInfo(**model_info)
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        if self.model_info.resize_id == "2e45058eeb09bd714122dcc5b6c5c22e":
            self.interpolation = transforms.InterpolationMode.BOX
        elif self.model_info.resize_id == "9b07f7038587b888bc1312559f48d163":
            self.interpolation = transforms.InterpolationMode.BILINEAR
        else:
            raise ValueError(f"Unknown resize_id: {self.model_info.resize_id}")
        self.size = (self.model_info.height, self.model_info.width)
        self.fps = 1 / self.model_info.period
        if self.model_info.offset != 0:
            raise ValueError(
                f"Offset must be 0, but got {self.model_info.offset}. "
                "This adapter only supports models with no offset."
            )

    def adapt(self, video: Video) -> torch.Tensor:
        # check if the aspect ratio is correct
        ratio = video.frames.shape[-2] / video.frames.shape[-1]
        if abs(ratio - self.size[0] / self.size[1]) > self.aspect_ratio_tolerance:
            raise ValueError(
                f"Aspect ratio mismatch: {ratio} != {self.size[0] / self.size[1]}"
            )
        transform_ls = []
        # resize the video frames
        transform_ls.append(
            transforms.Resize(size=self.size, interpolation=self.interpolation)
        )
        # if the video is color, convert to grayscale
        if video.frames.shape[0] == 3:
            transform_ls.append(
                transforms.Lambda(  # x is a tensor of shape (C, T, H, W)
                    lambda x: transforms.functional.rgb_to_grayscale(
                        x.permute(1, 0, 2, 3), num_output_channels=1
                    ).permute(1, 0, 2, 3)
                )
            )
        elif video.frames.shape[0] != 1:
            raise ValueError(
                f"Unsupported number of channels: {video.frames.shape[0]}. "
                "This adapter only supports grayscale or RGB videos."
            )

        adapt_transform = transforms.Compose(transform_ls)
        adapted_frames = adapt_transform(video.frames)

        # interpolate the temporal dimension only if the model fps is integer multiple of the video fps
        if self.fps != video.fps:
            if (
                self.fps.is_integer()
                and video.fps.is_integer()
                and self.fps % video.fps == 0
            ):
                num_frames = int(
                    video.frames.shape[1] * self.fps / video.fps
                )  # frames are in (C, T, H, W) format
                num_repeats = int(self.fps / video.fps)
                repeat_index = torch.arange(
                    0, video.frames.shape[1], step=1 / num_repeats
                ).long()

                # apply the transformations
                adapted_frames = adapted_frames[:, repeat_index, :, :].contiguous()
                adapted_video = Video(
                    frames=adapted_frames,
                    fps=self.fps,
                    metadata=video.metadata.iloc[repeat_index],
                )
            else:
                raise ValueError(
                    f"Model fps {self.fps} is not an integer multiple of video fps {video.fps}. "
                    "This adapter only supports models with fps that is an integer multiple of the video fps."
                )
        else:
            adapted_video = Video(
                frames=adapted_frames,
                fps=self.fps,
                metadata=video.metadata,
            )

        return adapted_video
