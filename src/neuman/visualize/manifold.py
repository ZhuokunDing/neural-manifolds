from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly import express as px
import base64
import io
from PIL import Image

from neuman.visualize.utils import TqdmWriter


def svd_flip(U, V):
    """
    Flip the signs of the singular vectors to ensure consistency.

    Parameters:
        U (numpy.ndarray): Left singular vectors.
        V (numpy.ndarray): Right singular vectors.

    Returns:
        tuple: Flipped U and V.
    """
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, np.arange(U.shape[1])])
    U *= signs
    V *= signs[:, np.newaxis]  # Ensure V is broadcastable
    return U, V


def create_pca_representation(responses, n_components=3, sklearn=True):
    """
    Create PCA representation of responses.

    Parameters:
        responses (np array): Model responses with shape (num_samples, num_features).

    Returns:
        np.ndarray: PCA-transformed responses with shape (num_samples, 3).
    """
    if isinstance(responses, torch.Tensor):
        responses = responses.cpu().numpy()
    if sklearn:
        pca = PCA(n_components=n_components)
        pca_responses = pca.fit_transform(responses.reshape(-1, responses.shape[-1]))
        variance_explained = pca.explained_variance_ratio_
    else:
        # implement PCA manually
        responses = responses - responses.mean(axis=0)  # center the data
        U, S, Vh = np.linalg.svd(responses, full_matrices=False)
        U, Vh = svd_flip(U, Vh)
        pca_responses = U[:, :n_components] @ np.diag(S[:n_components])
        variance_explained = (S[:n_components] ** 2) / np.sum(S**2)

    return pca_responses, variance_explained


def plot_manifold(
    pca_responses,
    c=None,
    ax=None,
    downsample_factor=1,
    s=0.1,
    alpha=0.7,
    cmap="turbo",
    category_mean=False,
    view_elev=30,
    view_azim=30,
):
    """
    Plot the PCA manifold in a 3D scatter plot.

    Parameters:
        pca_responses (np.ndarray): PCA-transformed responses with shape (num_samples, 3).
        c (pd.Series): Colors for the scatter plot. Should be ordered categories or continuous values.
        ax (matplotlib.axes._subplots.Axes3DSubplot): Optional axis to plot on.
        downsample_factor (float): Factor to downsample the data for faster plotting, between 0 and 1.
        s (float): Size of the scatter points.
        cmap (str): Colormap to use for the scatter plot.
        category_mean (bool): If True, plot the mean of each category, only relevant if `c` is provided and is a categorical variable.

    Returns:
        tuple: (scatter, ax) where scatter is the scatter plot object and ax is the axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=view_elev, azim=view_azim)
    if 0 < downsample_factor < 1:
        downsample_indices = slice(None, None, int(1 / downsample_factor))
    elif downsample_factor == 1:
        downsample_indices = slice(None)
    elif downsample_factor <= 0 or downsample_factor > 1:
        raise ValueError(
            "downsample_factor must be greater than 0 and less than or equal to 1"
        )
    scatter = ax.scatter(
        pca_responses[:, 0][downsample_indices],
        pca_responses[:, 1][downsample_indices],
        pca_responses[:, 2][downsample_indices],
        c=c[downsample_indices] if c is not None else np.arange(pca_responses.shape[0]),
        cmap=cmap,
        alpha=alpha,
        zorder=1,
        s=s,
    )
    if category_mean and c is not None:
        # c should be a categorical variable
        assert pd.api.types.is_categorical_dtype(c), "c must be a categorical variable"
        plotting_data = []
        for category in c.categories:
            category_mask = c == category
            mean_response = pca_responses[category_mask].mean(axis=0)
            plotting_data.append(
                dict(
                    x=mean_response[0],
                    y=mean_response[1],
                    z=mean_response[2],
                    color=scatter.cmap(scatter.norm(category)),
                    size=s * 50,
                )
            )
        plotting_df = pd.DataFrame(plotting_data)
        ax.scatter(
            plotting_df["x"],
            plotting_df["y"],
            plotting_df["z"],
            c=plotting_df["color"],
            s=plotting_df["size"],
            edgecolor="black",
            zorder=2,
            linewidth=0.5,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA Responses Over Time")
    return ax


def create_animation(
    frames,
    responses,
    c=None,
    fps=30,
    filepath=None,
    bitrate=1000,
    downsample_factor=1,
    s=0.1,  # Default size for scatter points
    cmap="turbo",
    category_mean=False,
    pixel_space=False,  # If True, include pixel-space manifold in animation
):
    """
    Create an animation showing the video and PCA manifold(s).

    Two modes:
    1. Single manifold (default): Video | Neural-space PCA manifold
    2. Dual manifold: Video | Pixel-space PCA | Neural-space PCA

    Parameters:
        frames (torch.Tensor): Video frames with shape (C, T, H, W).
        responses (torch.Tensor): Model responses with shape (num_samples, num_features).
        c (pd.Categorical): Optional categorical colors for points.
        fps (int): Frames per second for the animation.
        filepath (str): If provided, save the animation to this file.
        bitrate (int): Bitrate for the saved video.
        downsample_factor (float): Factor to downsample the data for faster plotting, between 0 and 1.
        s (float): Size of scatter points.
        cmap (str): Colormap for scatter plot.
        category_mean (bool): If True, plot category means as larger dots.
        pixel_space (bool): If True, creates dual manifold animation with both pixel-space
            and neural-space PCA. Pixel features are automatically derived from frames.
    """
    # Step 1: Create PCA representation of neural responses
    pca_responses, variance_explained = create_pca_representation(responses)

    # Check if dual manifold mode
    dual_manifold = pixel_space

    if dual_manifold:
        # Derive pixel-space features from frames
        # Extract frames and remove channel dimension: (C, T, H, W) → (T, H, W)
        if isinstance(frames, torch.Tensor):
            pixel_frames = frames[0].cpu().numpy()  # (T, H, W)
        else:
            pixel_frames = frames[0]  # Already numpy

        # Flatten spatial dimensions: (T, H, W) → (T, H*W)
        T, H, W = pixel_frames.shape
        pixel_features = pixel_frames.reshape(T, H * W)

        # Normalize to [0, 1] range
        pixel_features = pixel_features.astype(np.float32) / 255.0

        # Create pixel-space PCA
        pca_pixel, pixel_variance = create_pca_representation(pixel_features)
        fig = plt.figure(figsize=(18, 6))
    else:
        fig = plt.figure(figsize=(12, 6))

    # Subplot for video
    if dual_manifold:
        ax1 = fig.add_axes([0.02, 0.1, 0.12, 0.8])  # Narrower for 3-panel layout
    else:
        ax1 = fig.add_axes([0.05, 0.1, 0.15, 0.8])  # Original layout
    ax1.axis("off")
    if frames.shape[0] == 1:  # Grayscale
        img = ax1.imshow(frames[0, 0].numpy(), cmap="gray", interpolation="nearest")
    else:  # RGB
        img = ax1.imshow(frames[:, 0].numpy().transpose(1, 2, 0))
    ax1.set_title("Video Frames", fontsize=10)

    if dual_manifold:
        # Subplot for pixel-space PCA manifold (middle)
        ax2 = fig.add_axes([0.18, 0.1, 0.38, 0.8], projection="3d")
        plot_manifold(
            pca_pixel,
            c=c,
            ax=ax2,
            downsample_factor=downsample_factor,
            s=s,
            cmap=cmap,
            category_mean=category_mean,
        )
        ax2.set_title(
            "Pixel-Space PCA\n(Var: {:.1%})".format(pixel_variance.sum()),
            fontsize=10
        )

        # Create red dot for pixel-space
        (red_dot_pixel,) = ax2.plot(
            [], [], [], "ro",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=10,
        )

        # Subplot for neural-space PCA manifold (right)
        ax3 = fig.add_axes([0.60, 0.1, 0.38, 0.8], projection="3d")
        plot_manifold(
            pca_responses,
            c=c,
            ax=ax3,
            downsample_factor=downsample_factor,
            s=s,
            cmap=cmap,
            category_mean=category_mean,
        )
        ax3.set_title(
            "Neural-Space PCA\n(Var: {:.1%})".format(variance_explained.sum()),
            fontsize=10
        )

        # Create red dot for neural-space
        (red_dot_neural,) = ax3.plot(
            [], [], [], "ro",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=10,
        )
    else:
        # Original single manifold layout
        ax2 = fig.add_axes([0.25, 0.1, 0.7, 0.8], projection="3d")
        plot_manifold(
            pca_responses,
            c=c,
            ax=ax2,
            downsample_factor=downsample_factor,
            s=s,
            cmap=cmap,
            category_mean=category_mean,
        )
        ax2.set_title(
            "PCA Manifold Over Time\n(Variance Explained: {:.2f})".format(
                variance_explained.sum()
            )
        )

        # Create the red dot for the animation
        (red_dot,) = ax2.plot(
            [], [], [], "ro",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=10,
        )

    rotation_per_frame = 5.0 / fps

    def update(frame):
        # Update video frame
        if frames.shape[0] == 1:  # Grayscale
            img.set_array(frames[0, frame].numpy())
        else:  # RGB
            img.set_array(frames[:, frame].numpy().transpose(1, 2, 0))

        if dual_manifold:
            # Update both manifold view angles
            ax2.view_init(elev=30, azim=frame * rotation_per_frame)
            ax3.view_init(elev=30, azim=frame * rotation_per_frame)

            # Update pixel-space red dot
            red_dot_pixel.set_data([pca_pixel[frame, 0]], [pca_pixel[frame, 1]])
            red_dot_pixel.set_3d_properties([pca_pixel[frame, 2]])
            red_dot_pixel.set_zorder(100)

            # Update neural-space red dot
            red_dot_neural.set_data([pca_responses[frame, 0]], [pca_responses[frame, 1]])
            red_dot_neural.set_3d_properties([pca_responses[frame, 2]])
            red_dot_neural.set_zorder(100)

            return img, red_dot_pixel, red_dot_neural
        else:
            # Update the view angle for the 3D scatter plot
            ax2.view_init(elev=30, azim=frame * rotation_per_frame)

            # Update the red dot position
            red_dot.set_data([pca_responses[frame, 0]], [pca_responses[frame, 1]])
            red_dot.set_3d_properties([pca_responses[frame, 2]])
            red_dot.set_zorder(100)

            return img, red_dot

    ani = FuncAnimation(
        fig, update, frames=len(frames[0]), interval=1000 / fps, blit=True
    )
    if filepath:
        writer = TqdmWriter(fps=fps, bitrate=bitrate, total_frames=len(frames[0]))
        ani.save(filepath, writer=writer)
    else:
        return HTML(ani.to_jshtml())
