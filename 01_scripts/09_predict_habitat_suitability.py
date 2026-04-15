# ============================================================
# Script Name: 09_predict_habitat_suitability.py
# Purpose: Predict habitat suitability & generate map (poster-ready)
# ============================================================

import os
import joblib
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from rasterio.mask import mask
from matplotlib.patches import Polygon, Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pyproj import Geod
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"

predictor_stack = os.path.join(
    project_root,
    "01_data","02_processed","03_aligned_predictors",
    "predictor_stack.tif"
)

model_file = os.path.join(
    project_root,
    "03_models",
    "random_forest_model.pkl"
)

boundary_file = os.path.join(
    project_root,
    "01_data","01_raw","03_boundary",
    "Odisha.shp"
)

output_raster = os.path.join(
    project_root,
    "04_outputs","03_rasters",
    "great_egret_habitat_suitability.tif"
)

output_map = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "great_egret_habitat_suitability.png"
)


# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

model = joblib.load(model_file)


# ----------------------------------------------------------
# LOAD PREDICTOR STACK
# ----------------------------------------------------------

with rasterio.open(predictor_stack) as src:
    stack = src.read()
    meta = src.meta.copy()

rows, cols = stack.shape[1], stack.shape[2]
X = stack.reshape(stack.shape[0], -1).T


# ----------------------------------------------------------
# MODEL PREDICTION
# ----------------------------------------------------------

pred = model.predict_proba(X)[:, 1]
suitability = pred.reshape(rows, cols)


# ----------------------------------------------------------
# SMOOTH (OPTIONAL)
# ----------------------------------------------------------

suitability = gaussian_filter(suitability, sigma=1)


# ----------------------------------------------------------
# SAVE RASTER
# ----------------------------------------------------------

meta.update(count=1, dtype="float32")

with rasterio.open(output_raster, "w", **meta) as dst:
    dst.write(suitability.astype("float32"), 1)

print("Suitability raster saved")


# ----------------------------------------------------------
# LOAD BOUNDARY
# ----------------------------------------------------------

odisha = gpd.read_file(boundary_file).to_crs("EPSG:4326")


# ----------------------------------------------------------
# MASK RASTER
# ----------------------------------------------------------

with rasterio.open(output_raster) as src:
    clipped, transform = mask(src, odisha.geometry, crop=True, filled=False)

data = clipped[0]
data = np.where(data.mask, np.nan, data)


# ----------------------------------------------------------
# MAP STYLE
# ----------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(figsize=(8, 7), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())


# ----------------------------------------------------------
# RASTER EXTENT
# ----------------------------------------------------------

left = transform[2]
top = transform[5]
right = left + transform[0] * data.shape[1]
bottom = top + transform[4] * data.shape[0]

extent = [left, right, bottom, top]


# ----------------------------------------------------------
# PLOT SUITABILITY
# ----------------------------------------------------------

img = ax.imshow(
    data,
    extent=extent,
    cmap="viridis",
    origin="upper",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree(),
    zorder=1
)


# ----------------------------------------------------------
# BOUNDARY
# ----------------------------------------------------------

odisha.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8, zorder=3)


# ----------------------------------------------------------
# EXTENT
# ----------------------------------------------------------

minx, miny, maxx, maxy = odisha.total_bounds
buffer = 0.5
ax.set_extent([minx-buffer, maxx+buffer, miny-buffer, maxy+buffer])


# ----------------------------------------------------------
# LAT/LON TICKS
# ----------------------------------------------------------

ax.set_xticks(np.arange(82, 88, 2), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(18, 24, 2), crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(labelsize=10)


# ----------------------------------------------------------
# NORTH ARROW
# ----------------------------------------------------------

nx, ny = 0.85, 0.25
size_w, size_h = 0.04, 0.07

ax.add_patch(Polygon([[nx, ny], [nx-size_w/2, ny-size_h], [nx, ny-(size_h*0.7)]],
                     transform=ax.transAxes, facecolor="white", edgecolor="black", zorder=5))

ax.add_patch(Polygon([[nx, ny], [nx, ny-(size_h*0.7)], [nx+size_w/2, ny-size_h]],
                     transform=ax.transAxes, facecolor="black", edgecolor="black", zorder=5))

ax.text(nx, ny+0.01, "N", transform=ax.transAxes, ha="center",
        fontsize=10, weight="bold")


# ----------------------------------------------------------
# SCALE BAR
# ----------------------------------------------------------

geod = Geod(ellps="WGS84")

start_lon = maxx - 3
start_lat = miny + 0.15

segments = [40, 40, 80, 80, 80]
colors = ["black", "white", "black", "white", "black"]

current = start_lon
height = 0.03
positions = [start_lon]

for seg, color in zip(segments, colors):
    end, _, _ = geod.fwd(current, start_lat, 90, seg * 1000)

    ax.add_patch(Rectangle(
        (current, start_lat - height/2),
        end - current,
        height,
        transform=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor="black",
        zorder=6
    ))

    current = end
    positions.append(current)

labels = [0, 40, 80, 160, 240, 320]

for pos, label in zip(positions, labels):
    ax.text(pos, start_lat - 0.18, str(label),
            transform=ccrs.PlateCarree(), ha="center", fontsize=9)

ax.text(positions[-1] + 0.1, start_lat - 0.18, "km",
        transform=ccrs.PlateCarree(), fontsize=9)


# ----------------------------------------------------------
# COLORBAR (RIGHT SIDE - VERTICAL)
# ----------------------------------------------------------

cbar = plt.colorbar(
    img,
    ax=ax,
    orientation="vertical",
    fraction=0.035,
    pad=0.02
)

cbar.set_label("Habitat Suitability", fontsize=11)
cbar.ax.tick_params(labelsize=9)

cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

cbar.outline.set_linewidth(0.5)


# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------

plt.savefig(output_map, dpi=300, bbox_inches="tight")
plt.close()

print("Habitat suitability map saved")