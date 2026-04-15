# ============================================================
# Script Name: 14_calculate_habitat_area.py
# Location: D:\Project\01_research\02_scripts\
# Purpose: Calculate pixel count and area of suitable habitat
# Author: Yeshwant
# Created On: 15 March 2026
# ============================================================

import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile


# ==========================================================
# PROJECT ROOT
# ==========================================================

project_root = r"D:\Project\01_research"


# ==========================================================
# FILE PATHS
# ==========================================================

binary_raster_path = os.path.join(
    project_root,
    "04_outputs", "03_rasters",
    "great_egret_binary_habitat.tif"
)

boundary_file = os.path.join(
    project_root,
    "01_data", "01_raw", "03_boundary",
    "Odisha.shp"
)

output_table = os.path.join(
    project_root,
    "04_outputs", "01_tables",
    "great_egret_habitat_area.csv"
)

os.makedirs(os.path.dirname(output_table), exist_ok=True)


# ==========================================================
# SETTINGS
# ==========================================================

dst_crs = "EPSG:32645"   # UTM Zone 45N (meters)
nodata_value = 255       # background


# ==========================================================
# LOAD STUDY AREA
# ==========================================================

odisha = gpd.read_file(boundary_file).to_crs(dst_crs)
print("✅ Odisha boundary loaded")


# ==========================================================
# LOAD & REPROJECT RASTER
# ==========================================================

with rasterio.open(binary_raster_path) as src:

    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds
    )

    print("🔄 Reprojecting raster to UTM...")

    reprojected_array = np.full(
        (height, width),
        nodata_value,
        dtype=np.uint8
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=reprojected_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )


# ==========================================================
# CLIP TO ODISHA
# ==========================================================

with MemoryFile() as memfile:

    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        crs=dst_crs,
        transform=transform,
        dtype=np.uint8,
        nodata=nodata_value
    ) as dataset:

        dataset.write(reprojected_array, 1)

        clipped, clipped_transform = mask(
            dataset,
            odisha.geometry,
            crop=True,
            nodata=nodata_value
        )

        final_data = clipped[0]

print("✂️ Raster clipped to Odisha")


# ==========================================================
# PIXEL AREA CALCULATION
# ==========================================================

pixel_width = clipped_transform[0]
pixel_height = abs(clipped_transform[4])

pixel_area_km2 = (pixel_width * pixel_height) / 1_000_000

print(f"📏 Pixel area: {pixel_area_km2:.6f} km²")


# ==========================================================
# PIXEL COUNT (WITH NODATA HANDLING)
# ==========================================================

valid_pixels = final_data != nodata_value

suitable_pixels = np.sum((final_data == 1) & valid_pixels)
unsuitable_pixels = np.sum((final_data == 0) & valid_pixels)

total_pixels = suitable_pixels + unsuitable_pixels


# ==========================================================
# AREA CALCULATION
# ==========================================================

suitable_area_km2 = suitable_pixels * pixel_area_km2
unsuitable_area_km2 = unsuitable_pixels * pixel_area_km2
total_area_km2 = total_pixels * pixel_area_km2

percentage_suitable = (suitable_area_km2 / total_area_km2) * 100


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n" + "="*45)
print("        PIXEL COUNT RESULTS")
print("="*45)

print(f"Suitable Pixels (1):   {suitable_pixels:,}")
print(f"Unsuitable Pixels (0): {unsuitable_pixels:,}")
print(f"Total Pixels:          {total_pixels:,}")

print("="*45)

print("\n" + "="*45)
print("        HABITAT AREA RESULTS")
print("="*45)

print(f"Total Odisha Area:       {total_area_km2:,.2f} km²")
print(f"Suitable Habitat Area:   {suitable_area_km2:,.2f} km²")
print(f"Unsuitable Habitat Area: {unsuitable_area_km2:,.2f} km²")
print(f"Habitat Percentage:      {percentage_suitable:.2f}%")

print("="*45)


# ==========================================================
# SAVE OUTPUT
# ==========================================================

results = pd.DataFrame({

    "Species": ["Great Egret"],

    "Total_Odisha_km2": [round(total_area_km2, 2)],

    "Suitable_Habitat_km2": [round(suitable_area_km2, 2)],

    "Unsuitable_Habitat_km2": [round(unsuitable_area_km2, 2)],

    "Percentage_Suitable": [round(percentage_suitable, 2)],

    "Suitable_Pixels": [int(suitable_pixels)],

    "Unsuitable_Pixels": [int(unsuitable_pixels)],

    "Total_Pixels": [int(total_pixels)]
})

results.to_csv(output_table, index=False)

print("\n💾 Results saved to:")
print(output_table)


# ==========================================================
# END
# ==========================================================