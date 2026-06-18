# Datathon PDS07: Regression Problem - Datamecum

This repository contains the Jupyter Notebook with the solution to the regression problem (Datathon PDS07) developed for the Data Science Expert Program at Datamecum. This approach achieved fourth place in the competition.

The notebook is documented in English, Spanish, and German to facilitate reading.

## Problem Statement

The challenge involved predicting a target variable (`deseada`) using data from ten sensors (V0-V9). The primary complexity of this dataset is a simulated sensor failure: the most predictive variable (V0) is present in 100% of the training data but is missing in approximately 75% of the test data (82 out of 110 rows). 

## Methodology & Architecture

The development follows a structured process focused on handling the V0 distribution shift without injecting error through naive imputation. The final architecture is an ensemble (`BlendRegressor`) of three distinct approaches:

* **Two-Model Routing (`TwoModelV0Regressor`):** A custom scikit-learn compatible estimator that trains two separate base models (one with V0, one without). During inference, it routes rows with a valid V0 to the first model and rows with a missing V0 to the fallback model. Non-V0 missing values are handled via `IterativeImputer` (BayesianRidge).
* **Native NaN Handling:** A `HistGradientBoostingRegressor` model is included in the blend to process missing values natively, allowing the tree algorithm to learn the optimal split for missing data.
* **Honest Cross-Validation:** Standard cross-validation was replaced with a custom evaluation strategy. To accurately reflect the test set conditions and estimate the true Mean Squared Error (MSE), V0 is artificially masked in 74.5% of the rows in every validation fold.
* **Outlier Retention:** Extreme values were intentionally retained without clipping. This decision leverages the robustness of tree-based algorithms and ensures the model is exposed to the full data variance, which is critical when evaluated under the MSE metric.

## Technologies

* Python
* Pandas & NumPy
* Scikit-Learn (RandomForestRegressor, HistGradientBoostingRegressor, IterativeImputer)
* Matplotlib & Seaborn

## Author

**David Riquelme Riquelme**
[GitHub Profile](https://github.com/RiquelmeRiquelme)