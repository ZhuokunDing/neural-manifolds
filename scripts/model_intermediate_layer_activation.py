# %%
# load a SingleScan model
import os

os.environ.setdefault("DJ_HOST", "at-database.stanford.edu")
os.environ.setdefault("DJ_USER", "YOUR_USERNAME")  # Change this to your MySQL username
os.environ.setdefault("DJ_PASS", "YOUR_PASSWORD")  # Change this to your MySQL password

from neuman.models.foundation import SingleScan

MODEL_KEY = {
    "model_set_id": "20a103eeb9b508d08e593a6db0bb1d07",
    "data_id": "0a7a8be8de7e14c059f1fc39ba6ac933",
    "network_id": "c17d459afa99a88b3e48a32fbabc21e4",
    "instance_id": "15c03d50c410911ed4937feffbfebd95",
}

model = SingleScan(model_key=MODEL_KEY)

# %%
# print the names of the layers in the model
[m[0] for m in model.model.named_children()]

# %%
# let's focus on the core
[m[0] for m in model.model.core.named_children()]

# %%
# Generate an RDK video and adapt it for the model
from neuman.video.loader.rdk import RDKLoader
from neuman.video.adapter.foundation import FoundationAdapter
import torch

rdk_loader = RDKLoader(
    width=256,
    height=144,
    duration=3.33,
    fps=30.0,
    num_dots=200,
    coherence=1.0,
    rotation_speed=108,
    disk_radius=0.4,
    seed=42,
)

video = rdk_loader.load_video()

# Prepend N_LEADIN copies of the first RDK frame as a static lead-in.
# This warms up the LSTM with the same image content that will then start moving,
# isolating the network's response to motion. Lead-in metadata is NaN so these
# rows are easy to drop before PCA.
from neuman.video import Video
import numpy as np
import pandas as pd

N_LEADIN = 10
leadin_frames = video.frames[:, :1].expand(-1, N_LEADIN, -1, -1).clone()
combined_frames = torch.cat([leadin_frames, video.frames], dim=1)

leadin_metadata = pd.DataFrame({col: [np.nan] * N_LEADIN for col in video.metadata.columns})
combined_metadata = pd.concat([leadin_metadata, video.metadata], ignore_index=True)

video = Video(frames=combined_frames, metadata=combined_metadata, fps=video.fps)

adapter = FoundationAdapter(model.model_info)
adapted_video = adapter.adapt(video)

video.play()

# %%
# Register forward hooks on the recurrent (CvtLstm) cell.
# Visual.forward runs once per timestep, so each hook fires once per frame.
from collections import defaultdict
import torch

recurrent = model.model.core.recurrent

activations = defaultdict(list)


def out_hook(module, inputs, output):
    # output of CvtLstm: [N, O, H, W] -- post-projection, what readout sees
    activations["recurrent_out"].append(output.detach().cpu())


def state_hook(module, inputs, output):
    # raw LSTM states stashed in self.past at the end of CvtLstm.forward
    activations["h"].append(module.past["h"].detach().cpu())
    activations["c"].append(module.past["c"].detach().cpu())


handle_out = recurrent.register_forward_hook(out_hook)
handle_state = recurrent.register_forward_hook(state_hook)

# %%
# Run inference -- drives the per-timestep loop in generate_response
with torch.no_grad():
    responses = model(adapted_video.frames)

handle_out.remove()
handle_state.remove()

# Stack per-timestep entries into [T, ...] tensors
recurrent_out = torch.stack(activations["recurrent_out"], dim=0)
h_seq = torch.stack(activations["h"], dim=0)
c_seq = torch.stack(activations["c"], dim=0)

print(f"responses:     {tuple(responses.shape)}")
print(f"recurrent_out: {tuple(recurrent_out.shape)}  [T, N, O, H, W]")
print(f"h_seq:         {tuple(h_seq.shape)}  [T, N, S*hidden, H, W]")
print(f"c_seq:         {tuple(c_seq.shape)}  [T, N, S*hidden, H, W]")

# %%
# PCA manifolds: CvtLstm hidden state vs. final readout responses.
# Drop the lead-in timesteps from both so they don't influence the projections.
from neuman.visualize.manifold import create_pca_representation, plot_manifold
import pandas as pd
from matplotlib import pyplot as plt

rotation_post_leadin = adapted_video.metadata["rotation_angle_deg"].values[N_LEADIN:]
rotation_categories = pd.Categorical(rotation_post_leadin, ordered=True)

h_post = h_seq[N_LEADIN:].reshape(h_seq.shape[0] - N_LEADIN, -1)  # hidden state
r_post = responses[N_LEADIN:]                                     # readout output

h_pca, h_var = create_pca_representation(h_post, sklearn=True)
r_pca, r_var = create_pca_representation(r_post, sklearn=True)

for pca, var, title in [
    (h_pca, h_var, "CvtLstm hidden state"),
    (r_pca, r_var, "Readout responses"),
]:
    fig = plt.figure(figsize=(12, 10))
    plot_manifold(
        pca,
        c=rotation_categories,
        s=0.5,
        alpha=0.3,
        category_mean=True,
        view_azim=15,
        view_elev=15,
    )
    plt.title(
        f"{title} manifold\n(Variance Explained: {var.sum():.2f})",
        fontsize=14,
        pad=20,
    )
    plt.show()

# %%
