# ============================================================
# Script Name: 08_generate_occurrence_map.py
# Location: Project/01_research/02_scripts/
# Purpose: Generate spatial map of cleaned occurrence points
# Author: Yeshwant
# Created On: 06 March 2026
# ============================================================

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pyproj import Geod

# ----------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"

occurrence_file = os.path.join(
    project_root,
    "01_data","02_processed","01_cleaned_occurrence",
    "occurrence_thinned.csv"
)

boundary_file = os.path.join(
    project_root,
    "01_data","01_raw","03_boundary",
    "Odisha.shp"
)

output_map = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "great_egret_occurrence_odisha.png"
)

os.makedirs(os.path.dirname(output_map), exist_ok=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

occ_df = pd.read_csv(occurrence_file)

occ_gdf = gpd.GeoDataFrame(
    occ_df,
    geometry=gpd.points_from_xy(
        occ_df["decimallongitude"],
        occ_df["decimallatitude"]
    ),
    crs="EPSG:4326"
)

odisha = gpd.read_file(boundary_file).to_crs("EPSG:4326")

# ----------------------------------------------------------
# FONT STYLE
# ----------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

# ----------------------------------------------------------
# CREATE MAP
# ----------------------------------------------------------

fig = plt.figure(figsize=(8,8), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor("white")

# ----------------------------------------------------------
# PLOT STATE BOUNDARY
# ----------------------------------------------------------

odisha.boundary.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.8,
    zorder=1
)

# ----------------------------------------------------------
# PLOT OCCURRENCE POINTS
# ----------------------------------------------------------

occ_gdf.plot(
    ax=ax,
    color="#006400",
    markersize=8,
    zorder=2
)

# ----------------------------------------------------------
# MAP EXTENT
# ----------------------------------------------------------

minx, miny, maxx, maxy = odisha.total_bounds

buffer = 0.5

ax.set_extent([
    minx-buffer,
    maxx+buffer,
    miny-buffer,
    maxy+buffer
])

# ----------------------------------------------------------
# LAT/LON TICKS (LEFT + BOTTOM ONLY)
# ----------------------------------------------------------

ax.set_xticks(np.arange(82, 88, 2), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(18, 24, 2), crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(
    labelsize=10,
    direction="out",
    length=6,
    pad=8,
    top=False,
    right=False,
    labeltop=False,
    labelright=False
)

# ----------------------------------------------------------
# NORTH ARROW
# ----------------------------------------------------------

# nx=0.85 aligns it with the right side of the map
# ny=0.25 puts it safely above the scale bar area
nx, ny = 0.85, 0.25 
size_w = 0.04
size_h = 0.07

left_tri = Polygon(
    [[nx, ny], [nx-size_w/2, ny-size_h], [nx, ny-(size_h*0.7)]],
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="black",
    linewidth=1,
    zorder=5
)

right_tri = Polygon(
    [[nx, ny], [nx, ny-(size_h*0.7)], [nx+size_w/2, ny-size_h]],
    transform=ax.transAxes,
    facecolor="black",
    edgecolor="black",
    linewidth=1,
    zorder=5
)

ax.add_patch(left_tri)
ax.add_patch(right_tri)

ax.text(
    nx,
    ny+0.01,
    "N",
    transform=ax.transAxes,
    ha="center",
    weight="bold",
    fontsize=10
)

# ----------------------------------------------------------
# SCALE BAR (SEGMENTED)
# ----------------------------------------------------------

geod = Geod(ellps="WGS84")

start_lon = maxx - 3
start_lat = miny + 0.15

segments = [40,40,80,80,80]
colors = ["black","white","black","white","black"]

current_lon = start_lon
height = 0.03

for seg,color in zip(segments,colors):

    end_lon,end_lat,_ = geod.fwd(current_lon,start_lat,90,seg*1000)

    rect = Rectangle(
        (current_lon,start_lat-height/2),
        end_lon-current_lon,
        height,
        transform=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        zorder=6
    )

    ax.add_patch(rect)

    current_lon = end_lon

# ----------------------------------------------------------
# SCALE BAR LABELS
# ----------------------------------------------------------

label_distances = [0,40,80,160,240,320]

current_lon = start_lon
positions = [start_lon]

for dist in [40,40,80,80,80]:
    current_lon,_,_ = geod.fwd(positions[-1],start_lat,90,dist*1000)
    positions.append(current_lon)

for pos,lab in zip(positions,label_distances):

    ax.text(
        pos,
        start_lat - 0.15,
        f"{lab}",
        transform=ccrs.PlateCarree(),
        ha="center",
        fontsize=9
    )

ax.text(
    positions[-1] + 0.1,
    start_lat - 0.15,
    "km",
    transform=ccrs.PlateCarree(),
    fontsize=9
)

# ----------------------------------------------------------
# LEGEND (TOP LEFT)
# ----------------------------------------------------------

legend_elements = [
    Line2D(
        [0],[0],
        marker='o',
        color='w',
        label='Occurrence Records',
        markerfacecolor='#006400',
        markeredgewidth=0,
        markersize=8
    )
]

legend = ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(0.02,0.98),
    frameon=True,
    fancybox=False,
    facecolor="white",
    edgecolor="black",
    fontsize=10
)

legend.get_frame().set_linewidth(0.8)

# ----------------------------------------------------------
# SAVE MAP
# ----------------------------------------------------------

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Occurrence map saved to:", output_map)