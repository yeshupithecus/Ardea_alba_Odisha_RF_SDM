# ============================================================
# Script Name: 01_occurrence_cleaning_thinning.py
# Location: Project/01_research/02_scripts/
# Purpose: Clean and spatially thin GBIF occurrence records
# Author: Yeshwant
# Created On: 20 February 2026
# ============================================================

import os
import pandas as pd
import numpy as np
import yaml
from sklearn.neighbors import BallTree


# ------------------------------------------------------------
# Load configuration file
# ------------------------------------------------------------
config_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "05_config",
    "config.yaml"
)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# ------------------------------------------------------------
# Define file paths
# ------------------------------------------------------------
raw_occurrence_path = os.path.join(
    os.path.dirname(__file__),
    config["paths"]["raw_data"],
    "01_occurrence",
    config["data"]["occurrence_file"]
)

processed_folder = os.path.join(
    os.path.dirname(__file__),
    config["paths"]["processed_data"],
    "01_cleaned_occurrence"
)

os.makedirs(processed_folder, exist_ok=True)

cleaned_output_path = os.path.join(processed_folder, "occurrence_cleaned.csv")
thinned_output_path = os.path.join(processed_folder, "occurrence_thinned.csv")


# ------------------------------------------------------------
# Read GBIF occurrence file (tab-separated TXT)
# ------------------------------------------------------------
print("Reading occurrence data...")
df = pd.read_csv(raw_occurrence_path, sep="\t", low_memory=False)

# Standardize column names to lowercase
df.columns = df.columns.str.lower()


# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------
required_columns = ["species", "decimallatitude", "decimallongitude"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataset.")

df = df[required_columns]


# ------------------------------------------------------------
# Remove missing coordinates
# ------------------------------------------------------------
df = df.dropna(subset=["decimallatitude", "decimallongitude"])


# ------------------------------------------------------------
# Remove invalid coordinate ranges
# ------------------------------------------------------------
df = df[
    (df["decimallatitude"] >= -90) &
    (df["decimallatitude"] <= 90) &
    (df["decimallongitude"] >= -180) &
    (df["decimallongitude"] <= 180)
]


# ------------------------------------------------------------
# Remove duplicate coordinate records
# ------------------------------------------------------------
df = df.drop_duplicates(
    subset=["decimallatitude", "decimallongitude"]
)

print(f"Records after cleaning: {len(df)}")


# ------------------------------------------------------------
# Save cleaned occurrence file
# ------------------------------------------------------------
if config["outputs"]["save_cleaned_occurrence"]:
    df.to_csv(cleaned_output_path, index=False)
    print("Cleaned occurrence file saved.")


# ------------------------------------------------------------
# Spatial thinning using haversine distance
# ------------------------------------------------------------
print("Performing spatial thinning...")

coords = np.radians(
    df[["decimallatitude", "decimallongitude"]].values
)

tree = BallTree(coords, metric="haversine")

radius_km = config["model"]["thinning_distance_km"]
radius = radius_km / 6371  # Convert km to radians

indices = tree.query_radius(coords, r=radius)

keep_mask = np.ones(len(df), dtype=bool)

for i, neighbors in enumerate(indices):
    if keep_mask[i]:
        neighbors = neighbors[neighbors > i]
        keep_mask[neighbors] = False

df_thinned = df[keep_mask]

print(f"Records after thinning: {len(df_thinned)}")


# ------------------------------------------------------------
# Save thinned occurrence file
# ------------------------------------------------------------
if config["outputs"]["save_thinned_occurrence"]:
    df_thinned.to_csv(thinned_output_path, index=False)
    print("Thinned occurrence file saved.")


print("Occurrence cleaning and thinning completed successfully.")