from jupyter_dash import JupyterDash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from .image import PapyrusImage
from .volume import PapyrusVolume
import shutil
import os

DATA_PATH = "C:/Users/brett/OneDrive/Desktop/Vesuvius/vesuvius-challenge-ink-detection/"

# Remove all existing zarr files
for subdir in ["train", "test"]:
    for label in os.listdir(os.path.join(DATA_PATH, subdir)):
        path = os.path.join(DATA_PATH, subdir, label)
        for file in os.listdir(path):
            if file.endswith(".zarr"):
                #os.remove(os.path.join(path, file))
                pass

# Generate zarr files for all the images if they don't already exist
for subdir in ["train", "test"]:
    for label in os.listdir(os.path.join(DATA_PATH, subdir)):
        path = os.path.join(DATA_PATH, subdir, label)
        # Infrared image
        if subdir == "train" and not os.path.exists(os.path.join(path, "ir.zarr")):
            PapyrusImage.build_from_image(
                os.path.join(path, "ir.png"),
                os.path.join(path, "ir.zarr"),
            )
        # Data mask
        if not os.path.exists(os.path.join(path, "mask.zarr")):
            PapyrusImage.build_from_image(
                os.path.join(path, "mask.png"),
                os.path.join(path, "mask.zarr"),
            )
        # Truth set
        if subdir == "train" and not os.path.exists(os.path.join(path, "inklabels.zarr")):
            PapyrusImage.build_from_image(
                os.path.join(path, "inklabels.png"),
                os.path.join(path, "inklabels.zarr"),
            )
        # Subvolume stacks
        if not os.path.exists(os.path.join(path, "volume.zarr")):
            PapyrusVolume.build_from_tiffdir(
                os.path.join(path, "surface_volume"),
                os.path.join(path, "volume.zarr"),
            )

ZARRPATH = os.path.join(DATA_PATH, "train/1/ir.zarr")
image = PapyrusImage(ZARRPATH)

fig = go.Figure(layout={
    "autosize": True,
    "xaxis": {"autorange": True},
    "yaxis": {"autorange": "reversed", "scaleanchor": "x"},
    "dragmode": "pan",
    "height": 800,
    "width": 800,
})
array = image[:,:,3]
fig.add_trace(
    go.Heatmap(
        z = array,
        colorscale="gray",
        x0 = 0,
        dx = 0.01,
        y0 = 0,
        dy = 0.01,
        transpose=False,
        showscale=False,
    )
)
config={"scrollZoom": True}

app = JupyterDash()
app.layout = html.Div([dcc.Graph(figure=fig, config=config)])
app.run_server()