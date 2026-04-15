# ============================================================
# Script Name: 03_extract_environmental_values.py
# Location: Project/01_research/02_scripts/
# Purpose: Extract environmental variable values at occurrence locations
# Author: Yeshwant
# Created On: 27 February 2026
# ============================================================

import os
import pandas as pd
import rasterio
import numpy as np


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

occurrence_file = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "01_cleaned_occurrence",
    "occurrence_thinned.csv"
)

predictor_folder = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "03_aligned_predictors"
)

output_folder = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "04_model_dataset"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "environmental_dataset.csv"
)


# ----------------------------------------------------------
# LOAD OCCURRENCE DATA
# ----------------------------------------------------------

occ = pd.read_csv(occurrence_file)

occ.columns = occ.columns.str.lower()

print("Occurrence data loaded")
print("Total records:", len(occ))


# ----------------------------------------------------------
# DETECT COORDINATE COLUMNS
# ----------------------------------------------------------

if "decimallongitude" in occ.columns and "decimallatitude" in occ.columns:

    lon_col = "decimallongitude"
    lat_col = "decimallatitude"

elif "longitude" in occ.columns and "latitude" in occ.columns:

    lon_col = "longitude"
    lat_col = "latitude"

else:

    raise ValueError("No coordinate columns found in occurrence file")


# ----------------------------------------------------------
# CREATE COORDINATE LIST
# ----------------------------------------------------------

coords = list(zip(occ[lon_col], occ[lat_col]))

print("Coordinates prepared")


# ----------------------------------------------------------
# PREPARE DATAFRAME
# ----------------------------------------------------------

data = occ.copy()


# ----------------------------------------------------------
# LIST PREDICTOR RASTERS
# ----------------------------------------------------------

print("\nScanning predictor rasters...")

predictor_files = sorted([
    f for f in os.listdir(predictor_folder)
    if f.endswith("_aligned.tif")
])

print("Total predictors found:", len(predictor_files))


# ----------------------------------------------------------
# EXTRACT RASTER VALUES
# ----------------------------------------------------------

print("\nExtracting environmental variables...\n")

for file in predictor_files:

    raster_path = os.path.join(predictor_folder, file)

    variable_name = file.replace("_aligned.tif", "")

    print("Extracting:", variable_name)

    with rasterio.open(raster_path) as src:

        nodata = src.nodata

        # Faster sampling
        samples = list(src.sample(coords))

        values = []

        for val in samples:

            v = val[0]

            if nodata is not None and v == nodata:
                values.append(np.nan)
            else:
                values.append(v)

        data[variable_name] = values


# ----------------------------------------------------------
# REMOVE ROWS WITH NODATA
# ----------------------------------------------------------

data_clean = data.dropna()

print("\nRecords after removing nodata:", len(data_clean))


# ----------------------------------------------------------
# SAVE DATASET
# ----------------------------------------------------------

data_clean.to_csv(output_file, index=False)

print("\nEnvironmental dataset created successfully")
print("Saved at:")
print(output_file)