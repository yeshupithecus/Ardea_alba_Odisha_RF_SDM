# Ardea_alba_Odisha_RF_SDM
Here’s a **clean, minimal, professional README.md** (no extra clutter):

---

# Habitat Suitability of *Ardea alba* (Great Egret)

This project models the habitat suitability of *Ardea alba* in Odisha, India using a **Random Forest-based Species Distribution Model (SDM)**.

## Overview

The model combines species occurrence data with environmental variables to predict suitable habitats and understand key ecological drivers.

## Data

Occurrence data: GBIF
Environmental variables:

  * Annual precipitation (BIO12)
  * Elevation
  * NDVI
  * Distance to water

## Method

* Data cleaning and spatial thinning (5 km)
* Variable selection using VIF
* Pseudo-absence generation (1:3)
* Train-test split (80:20)
* Random Forest model training
* Evaluation using ROC-AUC

## Results

* Model performance: **AUC ≈ 0.75**
* High suitability: coastal areas, wetlands, river systems
* Suitable habitat: **~33% of study area**

## Structure

```
scripts/        # Model pipeline
data_sample/    # Small sample data
outputs/        # Maps and figures
config/         # Config and requirements
```

## Note

Raw environmental datasets are not included due to size.
Download from:

* [https://www.gbif.org/](https://www.gbif.org/)
* [https://www.worldclim.org/](https://www.worldclim.org/)
* [https://global-surface-water.appspot.com/](https://global-surface-water.appspot.com/)


## Run

```
pip install -r requirements.txt
python script_name.py
```


