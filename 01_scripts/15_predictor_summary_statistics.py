# ============================================================
# Script Name: 15_predictor_summary_statistics.py
# Location: Project/01_research/02_scripts/
# Purpose: Generate summary statistics of environmental predictors
# Author: Yeshwant
# Created On: 15 March 2026
# ============================================================

import os
import pandas as pd


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# INPUT DATASET
# ----------------------------------------------------------

dataset_file = os.path.join(
    project_root,
    "01_data","02_processed","04_model_dataset",
    "environmental_dataset.csv"
)


# ----------------------------------------------------------
# OUTPUT TABLE
# ----------------------------------------------------------

output_table = os.path.join(
    project_root,
    "04_outputs","01_tables",
    "predictor_summary_statistics.csv"
)

os.makedirs(os.path.dirname(output_table), exist_ok=True)


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv(dataset_file)

print("Dataset loaded")
print("Total records:", len(df))


# ----------------------------------------------------------
# FINAL PREDICTORS USED IN MODEL
# ----------------------------------------------------------

predictors = [

    "wc2.1_30s_bio_12",   # Annual precipitation
    "wc2.1_30s_elev",     # Elevation
    "distance_to_water",  # Distance to water
    "ndvi"                # Vegetation productivity

]


# ----------------------------------------------------------
# SELECT DATA
# ----------------------------------------------------------

data = df[predictors]


# ----------------------------------------------------------
# CALCULATE SUMMARY STATISTICS
# ----------------------------------------------------------

summary = data.describe().T

summary = summary.rename(columns={

    "min": "Minimum",
    "max": "Maximum",
    "mean": "Mean",
    "std": "Std_Dev"

})

summary = summary[["Minimum","Maximum","Mean","Std_Dev"]]

summary.reset_index(inplace=True)

summary = summary.rename(columns={"index":"Variable"})


# ----------------------------------------------------------
# SAVE TABLE
# ----------------------------------------------------------

summary.to_csv(output_table, index=False)

print("\nSummary statistics generated successfully")
print("Saved at:")
print(output_table)