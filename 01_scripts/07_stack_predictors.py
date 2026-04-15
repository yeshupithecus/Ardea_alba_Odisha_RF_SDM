# ============================================================
# Script Name: 07_stack_predictors.py
# Location: Project/01_research/02_scripts/
# Purpose: Stack all predictor rasters into a single multi-layer raster
# Author: Yeshwant
# Created On: 04 March 2026
# ============================================================

import os
import rasterio


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# INPUT PREDICTOR FOLDER
# ----------------------------------------------------------

predictor_folder = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "03_aligned_predictors"
)


# ----------------------------------------------------------
# OUTPUT STACK FILE
# ----------------------------------------------------------

output_stack = os.path.join(
    predictor_folder,
    "predictor_stack.tif"
)


# ----------------------------------------------------------
# FINAL PREDICTOR FILES
# ----------------------------------------------------------

predictor_files = [

    "wc2.1_30s_bio_12_aligned.tif",      # precipitation
    "wc2.1_30s_elev_aligned.tif",        # elevation
    "distance_to_water_aligned.tif",     # hydrology
    "ndvi_aligned.tif"                   # vegetation productivity

]


predictor_paths = [
    os.path.join(predictor_folder, f)
    for f in predictor_files
]


print("Predictors to stack:")
for p in predictor_paths:
    print(p)


# ----------------------------------------------------------
# READ METADATA FROM FIRST RASTER
# ----------------------------------------------------------

with rasterio.open(predictor_paths[0]) as src:

    meta = src.meta.copy()

    height = src.height
    width = src.width
    crs = src.crs
    transform = src.transform


meta.update(count=len(predictor_paths))


# ----------------------------------------------------------
# STACK RASTERS
# ----------------------------------------------------------

with rasterio.open(output_stack, "w", **meta) as dst:

    for i, path in enumerate(predictor_paths):

        with rasterio.open(path) as src:

            if (
                src.height != height or
                src.width != width or
                src.crs != crs
            ):
                raise ValueError(
                    f"Raster alignment mismatch: {path}"
                )

            data = src.read(1)

            dst.write(data, i + 1)

            print(f"Band {i+1} written: {path}")


print("\nPredictor stack created successfully")
print("Output:", output_stack)