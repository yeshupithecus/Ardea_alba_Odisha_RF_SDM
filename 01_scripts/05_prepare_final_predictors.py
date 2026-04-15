# ============================================================
# Script Name: 05_prepare_final_predictors.py
# Location: Project/01_research/02_scripts/
# Purpose: Select and prepare final set of predictors for modeling
# Author: Yeshwant
# Created On: 03 March 2026
# Updated On: 24 March 2026 (Correlation heatmap fixed)
# ============================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# INPUT DATASET
# ----------------------------------------------------------

input_file = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "04_model_dataset",
    "environmental_dataset.csv"
)


# ----------------------------------------------------------
# OUTPUT FILES
# ----------------------------------------------------------

output_dataset = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "04_model_dataset",
    "final_model_dataset.csv"
)

heatmap_output = os.path.join(
    project_root,
    "04_outputs",
    "02_figures",
    "final_correlation_heatmap.png"
)

vif_output = os.path.join(
    project_root,
    "04_outputs",
    "01_tables",
    "final_vif_scores.csv"
)

corr_table_output = os.path.join(
    project_root,
    "04_outputs",
    "01_tables",
    "final_correlation_matrix.csv"
)

# Ensure output directories exist
os.makedirs(os.path.dirname(vif_output), exist_ok=True)
os.makedirs(os.path.dirname(heatmap_output), exist_ok=True)


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset loaded")
print("Shape:", df.shape)


# ----------------------------------------------------------
# FINAL SELECTED PREDICTORS
# ----------------------------------------------------------

selected_predictors = [
    "wc2.1_30s_bio_12",   # Annual precipitation
    "wc2.1_30s_elev",     # Elevation
    "distance_to_water",  # Distance to wetlands
    "ndvi"                # Vegetation productivity
]


# ----------------------------------------------------------
# CREATE FINAL DATASET
# ----------------------------------------------------------

final_df = df[
    ["species", "decimallatitude", "decimallongitude"] +
    selected_predictors
]

final_df.to_csv(output_dataset, index=False)

print("Final modelling dataset saved at:", output_dataset)


# ----------------------------------------------------------
# CORRELATION MATRIX
# ----------------------------------------------------------

predictor_data = final_df[selected_predictors]

corr_matrix = predictor_data.corr()

# Save correlation matrix table
corr_matrix.to_csv(corr_table_output)
print("Correlation matrix saved at:", corr_table_output)

# Plot heatmap (FIXED SCALE)
plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(7, 5))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    vmin=-1,                 # FIX: force correct range
    vmax=1,
    center=0,                # FIX: center at zero
    linewidths=0.5,
    cbar_kws={"label": "Correlation coefficient"}
)

plt.title("Correlation Heatmap of Final Predictor Variables", fontsize=12)
plt.xlabel("Predictor Variables")
plt.ylabel("Predictor Variables")

plt.tight_layout()
plt.savefig(heatmap_output, dpi=300)

print("Correlation heatmap saved at:", heatmap_output)


# ----------------------------------------------------------
# VIF CALCULATION
# ----------------------------------------------------------

vif_data = pd.DataFrame()

vif_data["Variable"] = predictor_data.columns

vif_data["VIF"] = [
    variance_inflation_factor(predictor_data.values, i)
    for i in range(predictor_data.shape[1])
]

# Save VIF table
vif_data.to_csv(vif_output, index=False)

print("\nFinal VIF Scores:")
print(vif_data)
print("VIF table saved at:", vif_output)


# ----------------------------------------------------------
# COMPLETION MESSAGE
# ----------------------------------------------------------

print("\nScript 05 completed successfully")