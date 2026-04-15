# ============================================================
# Script Name: 02_prepare_environmental_rasters.py
# Location: Project/01_research/02_scripts/
# Purpose: Prepare, clip, and standardize environmental raster layers
# Author: Yeshwant
# Created On: 27 February 2026
# ============================================================

import os
import rasterio
import geopandas as gpd
import numpy as np

from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt


# ============================================================
# PROJECT ROOT
# ============================================================

project_root = r"D:\Project\01_research"


# ============================================================
# INPUT PATHS
# ============================================================

bioclim_folder = os.path.join(
    project_root,"01_data","01_raw","02_environmental","01_bioclimatic"
)

elevation_file = os.path.join(
    project_root,"01_data","01_raw","02_environmental","02_elevation",
    "wc2.1_30s_elev.tif"
)

# NDVI INPUT
ndvi_file = os.path.join(
    project_root,"01_data","01_raw","02_environmental","03_ndvi",
    "ndvi_5year_mean_odisha.tif"
)

water_tile_1 = os.path.join(
    project_root,"01_data","01_raw","02_environmental","04_surface_water",
    "extent_80E_20Nv1_4_2021.tif"
)

water_tile_2 = os.path.join(
    project_root,"01_data","01_raw","02_environmental","04_surface_water",
    "extent_80E_30Nv1_4_2021.tif"
)

boundary_file = os.path.join(
    project_root,"01_data","01_raw","03_boundary","Odisha.shp"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

clipped_folder = os.path.join(
    project_root,"01_data","02_processed","02_clipped_environmental"
)

aligned_folder = os.path.join(
    project_root,"01_data","02_processed","03_aligned_predictors"
)

os.makedirs(clipped_folder, exist_ok=True)
os.makedirs(aligned_folder, exist_ok=True)


# ============================================================
# LOAD BOUNDARY
# ============================================================

boundary = gpd.read_file(boundary_file)
geometry = boundary.geometry.values

print("Odisha boundary loaded")


# ============================================================
# CLIP FUNCTION
# ============================================================

def clip_raster(input_raster, output_raster):

    with rasterio.open(input_raster) as src:

        clipped_image, clipped_transform = mask(src, geometry, crop=True)

        meta = src.meta.copy()

        meta.update({
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform
        })

        with rasterio.open(output_raster,"w",**meta) as dst:
            dst.write(clipped_image)


# ============================================================
# CLIP BIOCLIM VARIABLES
# ============================================================

print("Clipping bioclim variables...")

for file in os.listdir(bioclim_folder):

    if file.endswith(".tif"):

        in_path = os.path.join(bioclim_folder,file)

        out_path = os.path.join(
            clipped_folder,
            file.replace(".tif","_clipped.tif")
        )

        clip_raster(in_path,out_path)

        print("Clipped:",file)


# ============================================================
# CLIP ELEVATION
# ============================================================

print("Clipping elevation...")

elev_out = os.path.join(clipped_folder,"wc2.1_30s_elev_clipped.tif")

clip_raster(elevation_file,elev_out)


# ============================================================
# CLIP NDVI
# ============================================================

print("Clipping NDVI...")

ndvi_clip = os.path.join(
    clipped_folder,
    "ndvi_clipped.tif"
)

clip_raster(ndvi_file,ndvi_clip)

print("NDVI clipped successfully")


# ============================================================
# PROCESS WATER TILES
# ============================================================

print("Processing water tiles...")

src1 = rasterio.open(water_tile_1)
src2 = rasterio.open(water_tile_2)

mosaic, transform = merge([src1,src2])

meta = src1.meta.copy()

meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform
})

temp_water = os.path.join(clipped_folder,"water_temp.tif")

with rasterio.open(temp_water,"w",**meta) as dst:
    dst.write(mosaic)

src1.close()
src2.close()

print("Clipping water...")

water_clipped = os.path.join(clipped_folder,"water_clipped.tif")

clip_raster(temp_water,water_clipped)

os.remove(temp_water)


# ============================================================
# ALIGN RASTERS
# ============================================================

print("Aligning rasters...")

reference = os.path.join(
    clipped_folder,
    "wc2.1_30s_bio_1_clipped.tif"
)

with rasterio.open(reference) as ref:

    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_width = ref.width
    ref_height = ref.height


for file in os.listdir(clipped_folder):

    if not file.endswith(".tif"):
        continue

    in_path = os.path.join(clipped_folder,file)

    out_path = os.path.join(
        aligned_folder,
        file.replace("_clipped.tif","_aligned.tif")
    )

    with rasterio.open(in_path) as src:

        meta = src.meta.copy()

        meta.update({
            "crs": ref_crs,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height
        })

        resampling_method = Resampling.bilinear

        with rasterio.open(out_path,"w",**meta) as dst:

            reproject(
                source=rasterio.band(src,1),
                destination=rasterio.band(dst,1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method
            )

        print("Aligned:",file)


# ============================================================
# FIX NDVI SCALING
# ============================================================

print("Checking NDVI scale...")

ndvi_aligned = os.path.join(aligned_folder,"ndvi_aligned.tif")

with rasterio.open(ndvi_aligned,"r+") as src:

    ndvi = src.read(1)

    max_val = np.nanmax(ndvi)

    # If NDVI > 1 it means it is scaled 0–10000
    if max_val > 1:

        print("NDVI detected as scaled 0–10000 → converting to 0–1")

        ndvi = ndvi / 10000.0

        src.write(ndvi.astype("float32"),1)

    else:

        print("NDVI already scaled correctly")


# ============================================================
# CREATE DISTANCE TO WATER
# ============================================================

print("Creating distance-to-water raster...")

water_aligned = os.path.join(aligned_folder,"water_aligned.tif")

distance_out = os.path.join(
    aligned_folder,
    "distance_to_water_aligned.tif"
)

with rasterio.open(water_aligned) as src:

    water = src.read(1)
    meta = src.meta.copy()

    water_binary = water > 0

    distance_pixels = distance_transform_edt(~water_binary)

    pixel_size_km = src.res[0] * 111

    distance_km = distance_pixels * pixel_size_km

    meta.update(dtype="float32")

    with rasterio.open(distance_out,"w",**meta) as dst:

        dst.write(distance_km.astype("float32"),1)

os.remove(water_aligned)

print("Distance-to-water raster created")

print("Script 02 completed successfully")