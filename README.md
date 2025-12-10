# Backup-Generator-Use-Prediction-using-Geo-Spatial-ML

## Project Description
Natural disasters such as hurricanes can severely damage the electric grid, producing widespread power loss that disrupts daily life and recovery efforts. In this study of Hurricane Ida, we used VIIRS Black Marble-derived power loss as a proxy for outage impact and applied a Random Forest classifier to predict backup generator presence at the grid-cell level using a mix of socioeconomic, built-environment, and environmental predictors.

Our results suggest that post-storm backup generator use is systematically associated with local conditions and can be meaningfully predicted from the features included in our model. Most importantly, the prominence of the Social Vulnerability Index (SVI) and the proportion of the Black population among high-inclusion predictors points to a clear equity dimension in backup generator access. These findings underscore the need for targeted preparedness and resilience policies that prioritize communities facing structural disadvantages in energy security and disaster recovery.

## Code Directory Structure
This repository contains the necessary scripts and notebooks to process geospatial environmental data, perform building classification, and execute predictive modeling. Other datasets mentioned in the report were manually downloaded from their respective websites.

### Core Data Processing & Modeling
* **post_process_black_marble_data.ipynb**: Processes the NASA Black Marble (VNP46A2/VNP46A3) suite to extract nighttime light (NTL) sub-datasets from HDF5 format and convert them into georeferenced TIFF files.
* **download_dem_data.ipynb**: A utility script to automate the retrieval of Digital Elevation Model (DEM) tiles from the USGS.
* **random_forest_model.ipynb**: The primary analytical pipeline that builds a Random Forest classifier to predict backup generator usage, performs variable inclusion proportion (VIP) analysis, and assesses feature correlation.

### Building Classification Pipeline
Located in `building_classification/`, this module classifies structures as residential/non-residential based on the methodology established by Arruda et al. (2024).

* **main.ipynb**: The primary execution notebook for structural classification.
* **map_buildings_original.py**: Implementation of the methodology using OpenStreetMap (OSM) building footprints and tags.
* **map_buildings_custom.py**: An adapted implementation for integrating custom footprint datasets (e.g., Microsoft Building Footprints) using OSM auxiliary tags.
* **utils.py**: Contains helper functions for coordinate system and Census shapefile management.
