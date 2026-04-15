# ============================================================
# Script Name: 04_check_multicollinearity.py
# Location: Project/01_research/02_scripts/
# Purpose: Assess multicollinearity among predictors using correlation analysis
# Author: Yeshwant
# Created On: 28 February 2026
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

dataset_file = os.path.join(
    project_root,
    "01_data",
    "02_processed",
    "04_model_dataset",
    "environmental_dataset.csv"
)


# ----------------------------------------------------------
# OUTPUT PATHS
# ----------------------------------------------------------

tables_folder = os.path.join(
    project_root,
    "04_outputs",
    "01_tables"
)

figures_folder = os.path.join(
    project_root,
    "04_outputs",
    "02_figures"
)

os.makedirs(tables_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


correlation_table = os.path.join(
    tables_folder,
    "correlation_matrix.csv"
)

vif_table = os.path.join(
    tables_folder,
    "vif_results.csv"
)

heatmap_output = os.path.join(
    figures_folder,
    "correlation_heatmap.png"
)


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv(dataset_file)

print("Dataset loaded")
print("Shape:", df.shape)


# ----------------------------------------------------------
# REMOVE NON-PREDICTOR COLUMNS
# ----------------------------------------------------------

predictors = df.drop(
    columns=[
        "species",
        "decimallatitude",
        "decimallongitude"
    ],
    errors="ignore"
)

print("Predictor variables:", predictors.shape[1])


# ----------------------------------------------------------
# CORRELATION MATRIX
# ----------------------------------------------------------

corr_matrix = predictors.corr()

corr_matrix.to_csv(correlation_table)

print("Correlation matrix saved")


# ----------------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(14,10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Correlation Heatmap of Environmental Predictors")

plt.tight_layout()

plt.savefig(
    heatmap_output,
    dpi=300
)

print("Correlation heatmap saved")


# ----------------------------------------------------------
# VIF CALCULATION
# ----------------------------------------------------------

vif_data = pd.DataFrame()

vif_data["Variable"] = predictors.columns

vif_data["VIF"] = [
    variance_inflation_factor(predictors.values, i)
    for i in range(predictors.shape[1])
]


# Sort variables by VIF
vif_data = vif_data.sort_values(
    by="VIF",
    ascending=False
)


# Save VIF table
vif_data.to_csv(vif_table, index=False)

print("\nVIF Results:")
print(vif_data)


print("\nVIF table saved")

print("\nScript 04 completed successfully")