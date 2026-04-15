# ============================================================
# Script Name: 09_predict_habitat_suitability.py
# Location: Project/01_research/02_scripts/
# Purpose: Predict habitat suitability across the study area
# Author: Yeshwant
# Created On: 09 March 2026
# ============================================================

import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from rasterio.mask import mask
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pyproj import Geod


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"

suitability_raster = os.path.join(
    project_root,
    "04_outputs","03_rasters",
    "great_egret_habitat_suitability.tif"
)

boundary_file = os.path.join(
    project_root,
    "01_data","01_raw","03_boundary",
    "Odisha.shp"
)

output_raster = os.path.join(
    project_root,
    "04_outputs","03_rasters",
    "great_egret_habitat_classified.tif"
)

output_map = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "Great_Egret_Habitat_Classification.png"
)


# ----------------------------------------------------------
# LOAD RASTER
# ----------------------------------------------------------

with rasterio.open(suitability_raster) as src:

    data = src.read(1)
    meta = src.meta.copy()


# ----------------------------------------------------------
# CLASSIFY HABITAT
# ----------------------------------------------------------

classified = np.zeros_like(data)

classified[(data > 0) & (data <= 0.33)] = 1
classified[(data > 0.33) & (data <= 0.66)] = 2
classified[data > 0.66] = 3


# ----------------------------------------------------------
# SAVE CLASSIFIED RASTER
# ----------------------------------------------------------

meta.update(
    dtype="uint8",
    count=1,
    nodata=0
)

with rasterio.open(output_raster,"w",**meta) as dst:
    dst.write(classified.astype("uint8"),1)

print("Classification raster saved")


# ----------------------------------------------------------
# LOAD BOUNDARY
# ----------------------------------------------------------

odisha = gpd.read_file(boundary_file).to_crs("EPSG:4326")


# ----------------------------------------------------------
# MASK RASTER
# ----------------------------------------------------------

with rasterio.open(output_raster) as src:

    clipped, transform = mask(
        src,
        odisha.geometry,
        crop=True,
        filled=False
    )

data = clipped[0]
data = np.where(data.mask,np.nan,data)


# ----------------------------------------------------------
# MAP STYLE
# ----------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(figsize=(8,8),dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())


# ----------------------------------------------------------
# EXTENT
# ----------------------------------------------------------

left = transform[2]
top = transform[5]

right = left + transform[0]*data.shape[1]
bottom = top + transform[4]*data.shape[0]

extent=[left,right,bottom,top]


# ----------------------------------------------------------
# COLOR MAP
# ----------------------------------------------------------

cmap = ListedColormap([
"#fff7bc",   # low suitability
"#7fc97f",   # moderate suitability
"#006400"    # high suitability
])


# ----------------------------------------------------------
# PLOT CLASSIFIED MAP
# ----------------------------------------------------------

img = ax.imshow(
    data,
    extent=extent,
    cmap=cmap,
    origin="upper",
    transform=ccrs.PlateCarree(),
    zorder=1
)


# ----------------------------------------------------------
# BOUNDARY
# ----------------------------------------------------------

odisha.boundary.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.8,
    zorder=3
)


# ----------------------------------------------------------
# MAP EXTENT
# ----------------------------------------------------------

minx,miny,maxx,maxy = odisha.total_bounds
buffer = 0.5

ax.set_extent([
    minx-buffer,
    maxx+buffer,
    miny-buffer,
    maxy+buffer
])


# ----------------------------------------------------------
# LAT / LON TICKS
# ----------------------------------------------------------

ax.set_xticks(np.arange(82,88,2),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(18,24,2),crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(labelsize=10)


# ----------------------------------------------------------
# NORTH ARROW
# ----------------------------------------------------------

nx,ny = 0.85,0.25
size_w = 0.04
size_h = 0.07

left_tri = Polygon(
[[nx,ny],[nx-size_w/2,ny-size_h],[nx,ny-(size_h*0.7)]],
transform=ax.transAxes,
facecolor="white",
edgecolor="black"
)

right_tri = Polygon(
[[nx,ny],[nx,ny-(size_h*0.7)],[nx+size_w/2,ny-size_h]],
transform=ax.transAxes,
facecolor="black",
edgecolor="black"
)

ax.add_patch(left_tri)
ax.add_patch(right_tri)

ax.text(nx,ny+0.01,"N",
transform=ax.transAxes,
ha="center",
fontsize=10,
weight="bold")


# ----------------------------------------------------------
# SCALE BAR
# ----------------------------------------------------------

geod = Geod(ellps="WGS84")

start_lon = maxx-3
start_lat = miny+0.15

segments=[40,40,80,80,80]
colors=["black","white","black","white","black"]

current=start_lon
height=0.03
positions=[start_lon]

for seg,color in zip(segments,colors):

    end,_,_=geod.fwd(current,start_lat,90,seg*1000)

    rect=Rectangle(
        (current,start_lat-height/2),
        end-current,
        height,
        transform=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor="black"
    )

    ax.add_patch(rect)

    current=end
    positions.append(current)


labels=[0,40,80,160,240,320]

for pos,label in zip(positions,labels):

    ax.text(
        pos,
        start_lat-0.18,
        str(label),
        transform=ccrs.PlateCarree(),
        ha="center",
        fontsize=9
    )

ax.text(
    positions[-1]+0.1,
    start_lat-0.18,
    "km",
    transform=ccrs.PlateCarree(),
    fontsize=9
)


# ----------------------------------------------------------
# LEGEND
# ----------------------------------------------------------

from matplotlib.lines import Line2D

legend_elements = [

Line2D([0],[0],color="#fff7bc",lw=6,label="Low suitability"),
Line2D([0],[0],color="#7fc97f",lw=6,label="Moderate suitability"),
Line2D([0],[0],color="#006400",lw=6,label="High suitability")

]

legend = ax.legend(
handles=legend_elements,
loc="upper left",
bbox_to_anchor=(0.02,0.98),
frameon=True,
fancybox=False,   # sharp rectangle edges
facecolor="white",
edgecolor="black",
fontsize=11,
handlelength=1.8,
borderpad=0.8
)

legend.get_frame().set_linewidth(1)


# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------

plt.savefig(output_map,dpi=300,bbox_inches="tight")
plt.close()

print("Habitat classification map saved")