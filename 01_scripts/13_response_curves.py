# ============================================================
# Script Name: 13_response_curves.py
# Location: Project/01_research/02_scripts/
# Purpose: Generate response curves for key environmental predictors
# Author: Yeshwant
# Created On: 12 March 2026
# ============================================================

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import PartialDependenceDisplay


# ==========================================================
# PROJECT ROOT
# ==========================================================

project_root = r"D:\Project\01_research"


# ==========================================================
# PATHS
# ==========================================================

dataset_file = os.path.join(
    project_root,
    "01_data","02_processed","04_model_dataset",
    "final_model_dataset.csv"
)

model_file = os.path.join(
    project_root,
    "03_models",
    "random_forest_model.pkl"
)

output_plot = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "great_egret_response_curves.png"
)

os.makedirs(os.path.dirname(output_plot), exist_ok=True)


# ==========================================================
# FINAL PREDICTOR VARIABLES (FIXED ORDER)
# ==========================================================

predictor_cols = [

    "wc2.1_30s_bio_12",
    "wc2.1_30s_elev",
    "distance_to_water",
    "ndvi"

]


# ==========================================================
# AXIS LABEL RENAMING
# ==========================================================

name_mapping = {

    "wc2.1_30s_bio_12": "Annual Precipitation (mm)",
    "wc2.1_30s_elev": "Elevation (m)",
    "distance_to_water": "Distance to Water (km)",
    "ndvi": "NDVI"

}


# ==========================================================
# PLOT STYLE
# ==========================================================

sns.set_style("white")

plt.rcParams.update({

    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11

})


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(dataset_file)

X = df[predictor_cols]

print("Predictors used:", predictor_cols)


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(model_file)

print("Random Forest model loaded")


# ==========================================================
# CREATE FIGURE
# ==========================================================

fig, axes = plt.subplots(

    2,
    2,
    figsize=(10,8),
    dpi=300,
    constrained_layout=True

)

axes = axes.flatten()


# ==========================================================
# PARTIAL DEPENDENCE PLOTS
# ==========================================================

display = PartialDependenceDisplay.from_estimator(

    estimator=model,
    X=X,
    features=predictor_cols,
    grid_resolution=50,
    ax=axes

)


# ==========================================================
# CLEAN STYLE
# ==========================================================

for i, ax in enumerate(axes):

    sns.despine(ax=ax)

    predictor = predictor_cols[i]

    new_label = name_mapping[predictor]

    ax.set_xlabel(new_label, fontfamily="serif", fontweight="bold")
    ax.set_ylabel("Partial Dependence", fontfamily="serif")

    for tick in (ax.get_xticklabels() + ax.get_yticklabels()):
        tick.set_fontfamily("serif")


# ==========================================================
# SAVE
# ==========================================================

plt.savefig(

    output_plot,
    dpi=300,
    bbox_inches="tight"

)

plt.close()

print("\nResponse curves saved to:")
print(output_plot)