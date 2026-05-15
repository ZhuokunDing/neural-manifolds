import datajoint as dj
import numpy as np
import pandas as pd
import torch
from torch import nn
from neuman.models.base import BaseVideoModel
from foundation.fnn import data, model
from foundation.virtual import fnn, recording, stimulus, utility
from djutils import merge


fc = dj.create_virtual_module("fc", "microns_funconnect_v0_0_2")

SET_KEY = {
    "model_set_id": "20a103eeb9b508d08e593a6db0bb1d07",
}

SINGLE_KEY = {
    "model_set_id": "20a103eeb9b508d08e593a6db0bb1d07",
    "data_id": "0a7a8be8de7e14c059f1fc39ba6ac933",
    "network_id": "c17d459afa99a88b3e48a32fbabc21e4",
    "instance_id": "15c03d50c410911ed4937feffbfebd95",
}


def load_single_scan_model(model_key: dict = SINGLE_KEY) -> tuple:
    # model info
    period = (data.Data & model_key).link.compute.sampling_period
    offset = (data.Data & model_key).link.compute.unit_offset
    height, width = (data.Data & model_key).link.compute.resolution
    resize_id = (data.Data & model_key).link.compute.resize_id
    model_info = {
        "period": period,
        "offset": offset,
        "height": height,
        "width": width,
        "resize_id": resize_id,
    }

    # neuron info
    neuron_info = {}
    data_scan_spec = fnn.Data.VisualScan.proj(
        "spec_id",
        "trace_filterset_id",
        "pipe_version",
        "animal_id",
        "session",
        "scan_idx",
        "segmentation_method",
        "spike_method",
    ) * fnn.Spec.VisualSpec.proj(
        "rate_id", offset_id="offset_id_unit", resample_id="resample_id_unit"
    )
    all_unit_trace_rel = (
        (fnn.Model & model_key)
        * data_scan_spec  # data_id -> specs + scan key
        * recording.ScanUnitOrder  # scan key + trace_filterset_id -> trace_ids
        * recording.Trace.ScanUnit  # trace_id -> unit key
    )
    all_units_df = all_unit_trace_rel.fetch(format="frame").reset_index()
    # fetch cc_max
    cc_max = (
        (recording.VisualMeasure & utility.Measure.CCMax & all_unit_trace_rel)
        .fetch(format="frame")
        .reset_index()
        .rename(columns={"measure": "cc_max"})
    )
    # fetch cc_abs
    cc_abs_df = (
        ((fnn.VisualRecordingCorrelation & utility.Correlation.CCSignal) & model_key)
        .fetch(format="frame")
        .reset_index()
        .rename(columns={"correlation": "cc_abs", "unit": "trace_order"})
    )  # this fetch is very slow
    # compute cc_norm
    neuron_info = (
        all_units_df.merge(cc_abs_df, how="left", validate="one_to_one")
        .merge(cc_max, how="left", validate="one_to_one")
        .assign(cc_norm=lambda df: df.cc_abs / df.cc_max)
    )

    # load model
    m = (model.Model & model_key).model(device="cuda")
    return m, neuron_info, model_info


# %%
class SingleScan(BaseVideoModel):
    def __init__(self, model_key: dict = SINGLE_KEY):
        """
        Initialize a SingleScan model for a specific scan.

        Args:
            model_key (dict): Key identifying the model in the database
        """
        super().__init__()
        self.model_key = model_key
        self.model, self.neuron_info, self.model_info = load_single_scan_model(
            model_key
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            videos (torch.Tensor): Input videos of shape (batch_size, channels, time, height, width)
        Returns:
            torch.Tensor: Model predictions of shape (batch_size, n_neurons, time)
        """
        # Ensure videos are on the same device as the model
        assert videos.dtype == torch.uint8, "Input must be of type uint8"
        videos = (
            videos.cpu().numpy()  # Convert to numpy int8 for the model
        )  # TODO: remove this line when the model supports torch input
        videos = videos.squeeze()
        # Forward pass through the model
        generator = self.model.generate_response(
            videos, perspectives=None, modulations=None, training=True, reset=True
        )  # training=True is set to return cuda tensors
        return torch.stack([*generator])

    def neuron_info(self) -> pd.DataFrame:
        """
        Get metadata about the neurons in this scan.

        Returns:
            pd.DataFrame: Metadata about the neurons
        """
        return self.neuron_info

    def model_info(self) -> dict:
        """
        Get metadata about the model.

        Returns:
            pd.DataFrame: Metadata about the model
        """
        return self.model_info


# TODO: Implement ModelSet to load all models trained for the MICrONS dataset
# class ModelSet(BaseVideoModel):
#     pass
