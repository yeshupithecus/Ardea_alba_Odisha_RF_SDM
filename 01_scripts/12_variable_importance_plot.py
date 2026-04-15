# ============================================================
# Script Name: 12_variable_importance_plot.py
# Location: Project/01_research/02_scripts/
# Purpose: Plot importance of environmental variables from the model
# Author: Yeshwant
# Created On: 10 March 2026
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

model_file = os.path.join(
    project_root,
    "03_models",
    "random_forest_model.pkl"
)

output_plot = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "great_egret_variable_importance.png"
)

os.makedirs(os.path.dirname(output_plot), exist_ok=True)


# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

model = joblib.load(model_file)

print("Random Forest model loaded")


# ----------------------------------------------------------
# VARIABLE NAMES (MODEL ORDER)
# ----------------------------------------------------------

variables = [

    "Annual precipitation (BIO12)",
    "Elevation (m)",
    "Distance to water (km)",
    "NDVI"

]


# ----------------------------------------------------------
# IMPORTANCE VALUES
# ----------------------------------------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Variable": variables,
    "Importance": importance

})


# ----------------------------------------------------------
# SORT VARIABLES (MOST IMPORTANT LAST)
# ----------------------------------------------------------

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ----------------------------------------------------------
# PLOT STYLE
# ----------------------------------------------------------

plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(7,5), dpi=300)


# ----------------------------------------------------------
# BAR PLOT
# ----------------------------------------------------------

plt.barh(

    importance_df["Variable"],
    importance_df["Importance"],
    color="#2c7fb8"

)


# ----------------------------------------------------------
# LABELS
# ----------------------------------------------------------

plt.xlabel("Variable Importance")
plt.ylabel("Environmental Predictor")

plt.title(
    "Variable Importance in Random Forest SDM\nGreat Egret"
)


# ----------------------------------------------------------
# SAVE FIGURE
# ----------------------------------------------------------

plt.tight_layout()

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Variable importance plot saved:")
print(output_plot)