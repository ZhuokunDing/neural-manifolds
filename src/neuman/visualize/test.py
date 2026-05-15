import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import torch
import numpy as np
from sklearn.decomposition import PCA
import base64
from PIL import Image
import io

# ---- Dummy Data ----
T, C, H, W = 100, 1, 64, 64
N = 50
frames = torch.rand(C, T, H, W)
response = torch.randn(T, N)

# ---- PCA ----
pca = PCA(n_components=3)
manifold = pca.fit_transform(response.numpy())  # [T, 3]


# ---- Pre-render Video Frames to Base64 ----
def encode_frame(frame_tensor):
    img = (
        (frame_tensor - frame_tensor.min())
        / (frame_tensor.max() - frame_tensor.min())
        * 255
    ).astype(np.uint8)
    im = Image.fromarray(img)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


frame_images = [encode_frame(frames[:, t, :, :].squeeze(0).numpy()) for t in range(T)]


# ---- Pre-render Static, Non-interactive Raster Plot ----
def create_static_raster_fig():
    fig = go.Figure()
    for i in range(response.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(T),
                y=response[:, i],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
        title="Neural Population Raster",
        xaxis_title="Time",
        yaxis_title="Response",
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    fig.update_traces(hoverinfo="skip")
    return fig


raster_base_fig = create_static_raster_fig()

# ---- Initialize App ----
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H3("Neural Manifold Visualizer"),
        # Video Frame
        html.Div(
            [
                html.Img(id="video-frame", style={"width": "256px", "height": "256px"}),
            ],
            style={"textAlign": "center"},
        ),
        # Static Raster Plot
        dcc.Graph(
            id="raster-plot", figure=raster_base_fig, config={"staticPlot": True}
        ),
        # 3D PCA Plot
        dcc.Graph(id="pca-plot"),
        # Play Button and Time Slider
        html.Div(
            [
                html.Button(
                    "▶️ Play",
                    id="play-button",
                    n_clicks=0,
                    style={"marginRight": "20px"},
                ),
                html.Div(
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=T - 1,
                        value=0,
                        step=1,
                        marks={0: "0", T // 2: f"{T // 2}", T - 1: f"{T - 1}"},
                        updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    style={"flex": 1},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "padding": "10px 40px"},
        ),
        # Timer
        dcc.Interval(
            id="interval",
            interval=33,  # 30 Hz
            n_intervals=0,
            disabled=True,
        ),
    ]
)


# ---- Callback: Update Frame and PCA Plot ----
@app.callback(
    Output("video-frame", "src"),
    Output("pca-plot", "figure"),
    Input("time-slider", "value"),
)
def update_visuals(t):
    frame_src = frame_images[t]

    pca_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=manifold[:, 0],
                y=manifold[:, 1],
                z=manifold[:, 2],
                mode="lines+markers",
                marker=dict(size=3, color="lightblue"),
                line=dict(width=2, color="gray"),
                name="Trajectory",
            ),
            go.Scatter3d(
                x=[manifold[t, 0]],
                y=[manifold[t, 1]],
                z=[manifold[t, 2]],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="Current Time",
            ),
        ]
    )
    pca_fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D PCA of Neural Manifold",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
    )

    return frame_src, pca_fig


# ---- Callback: Play/Pause Button ----
@app.callback(
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks):
    if n_clicks % 2 == 1:
        return False, "⏸️ Pause"
    else:
        return True, "▶️ Play"


# ---- Callback: Advance Time with Interval ----
@app.callback(
    Output("time-slider", "value"),
    Input("interval", "n_intervals"),
    State("time-slider", "value"),
)
def advance_time(_, current_t):
    return (current_t + 1) % T  # Loop


if __name__ == "__main__":
    app.run(debug=True, port=8050)
