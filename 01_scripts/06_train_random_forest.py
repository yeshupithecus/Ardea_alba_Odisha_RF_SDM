# ============================================================
# Script Name: 06_train_random_forest.py
# Location: Project/01_research/02_scripts/
# Purpose: Train Random Forest model using occurrence and predictor data
# Author: Yeshwant
# Created On: 03 March 2026
# ============================================================

import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt

from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

project_root = r"D:\Project\01_research"


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

dataset_file = os.path.join(
    project_root,
    "01_data","02_processed","04_model_dataset",
    "final_model_dataset.csv"
)

predictor_folder = os.path.join(
    project_root,
    "01_data","02_processed","03_aligned_predictors"
)

boundary_file = os.path.join(
    project_root,
    "01_data","01_raw","03_boundary",
    "Odisha.shp"
)

model_output = os.path.join(
    project_root,
    "03_models",
    "random_forest_model.pkl"
)

roc_output = os.path.join(
    project_root,
    "04_outputs","02_figures",
    "roc_curve.png"
)

os.makedirs(os.path.dirname(model_output), exist_ok=True)
os.makedirs(os.path.dirname(roc_output), exist_ok=True)


# ----------------------------------------------------------
# LOAD PRESENCE DATA
# ----------------------------------------------------------

df = pd.read_csv(dataset_file)

df["presence"] = 1

print("Presence records:", len(df))


# ----------------------------------------------------------
# LOAD STUDY AREA
# ----------------------------------------------------------

boundary = gpd.read_file(boundary_file)
boundary = boundary.to_crs("EPSG:4326")

minx, miny, maxx, maxy = boundary.total_bounds


# ----------------------------------------------------------
# PREPARE PRESENCE POINTS
# ----------------------------------------------------------

presence_points = [

    Point(xy) for xy in zip(
        df["decimallongitude"],
        df["decimallatitude"]
    )

]


# ----------------------------------------------------------
# GENERATE PSEUDO-ABSENCE POINTS (BUFFERED)
# ----------------------------------------------------------

print("Generating pseudo-absence points...")

n_absence = len(df) * 3

buffer_distance = 0.05   # ≈5 km

points = []

np.random.seed(42)

while len(points) < n_absence:

    x = np.random.uniform(minx, maxx)
    y = np.random.uniform(miny, maxy)

    candidate = Point(x, y)

    if not boundary.contains(candidate).any():
        continue

    too_close = False

    for p in presence_points:

        if candidate.distance(p) < buffer_distance:
            too_close = True
            break

    if not too_close:
        points.append((x, y))


pseudo_df = pd.DataFrame(

    points,
    columns=["decimallongitude","decimallatitude"]

)

pseudo_df["presence"] = 0

print("Pseudo-absence points generated:", len(pseudo_df))


# ----------------------------------------------------------
# EXTRACT ENVIRONMENTAL VALUES
# ----------------------------------------------------------

coords = list(zip(

    pseudo_df["decimallongitude"],
    pseudo_df["decimallatitude"]

))


predictors = [

    "wc2.1_30s_bio_12_aligned.tif",
    "wc2.1_30s_elev_aligned.tif",
    "distance_to_water_aligned.tif",
    "ndvi_aligned.tif"

]


for raster_name in predictors:

    raster_path = os.path.join(predictor_folder, raster_name)

    variable_name = raster_name.replace("_aligned.tif","")

    print("Extracting:", variable_name)

    with rasterio.open(raster_path) as src:

        values = [v[0] for v in src.sample(coords)]

        pseudo_df[variable_name] = values


pseudo_df = pseudo_df.dropna()


# ----------------------------------------------------------
# PREPARE PRESENCE DATA
# ----------------------------------------------------------

presence_df = df.drop(columns=["species"])


# ----------------------------------------------------------
# COMBINE DATA
# ----------------------------------------------------------

model_data = pd.concat([presence_df, pseudo_df])

model_data = model_data.dropna()

print("Total modelling records:", model_data.shape)


# ----------------------------------------------------------
# SELECT PREDICTORS
# ----------------------------------------------------------

predictor_cols = [

    "wc2.1_30s_bio_12",
    "wc2.1_30s_elev",
    "distance_to_water",
    "ndvi"

]


X = model_data[predictor_cols]

y = model_data["presence"]


# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# ----------------------------------------------------------
# RANDOM FOREST MODEL
# ----------------------------------------------------------

rf = RandomForestClassifier(

    n_estimators=1000,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42

)

rf.fit(X_train, y_train)

print("Random Forest model trained")


# ----------------------------------------------------------
# CROSS VALIDATION
# ----------------------------------------------------------

scores = cross_val_score(

    rf,
    X,
    y,
    cv=5,
    scoring="roc_auc"

)

print("Cross-validation AUC:", scores)
print("Mean AUC:", scores.mean())


# ----------------------------------------------------------
# TEST AUC
# ----------------------------------------------------------

y_pred = rf.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_pred)

print("Test AUC:", auc)


# ----------------------------------------------------------
# ROC CURVE
# ----------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest SDM")

plt.legend()

plt.savefig(roc_output, dpi=300)

plt.close()

print("ROC curve saved:", roc_output)


# ----------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------

importance = pd.Series(

    rf.feature_importances_,
    index=predictor_cols

)

print("\nFeature importance:")
print(importance.sort_values(ascending=False))


# ----------------------------------------------------------
# SAVE MODEL
# ----------------------------------------------------------

joblib.dump(rf, model_output)

print("Model saved:", model_output)