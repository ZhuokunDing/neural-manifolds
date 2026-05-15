import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from neuman.video import Video


class BaseLoader(ABC):
    @property
    @abstractmethod
    def load_video(self) -> Video:  # (C, T, H, W)
        """
        Abstract property to get the loaded video.
        """
        pass
