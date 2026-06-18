# Datathon PDS07: Regression Problem - Datamecum

This repository contains the Jupyter Notebook with the solution to the regression problem (Datathon PDS07) developed for the Data Science Expert Program at Datamecum.

To facilitate reading, the document is commented in three languages (English, Spanish, and German).

## Methodology

The development follows a structured process focused on avoiding data leakage and maximizing robustness against outliers:

*   **Exploration and Cleaning:** Distribution analysis and null detection. A technical decision was made to keep the outliers to preserve the natural variance of the sensor data.
*   **Preprocessing Pipeline:** Strict separation of training and test sets before applying transformations. The pipeline includes the creation of missing indicators, winsorization, scaling (StandardScaler), and imputation using k-Nearest Neighbors (KNNImputer).
*   **Modeling and Optimization:** Training robust models (Random Forest and Gradient Boosting) by systematically tuning hyperparameters using `GridSearchCV` with cross-validation (5-fold).
*   **Feature Selection:** Evaluation through Permutation Importance to detect and remove noisy variables confirmed by both models.
*   **Final Prediction:** Ensemble (average) of the Random Forest and Gradient Boosting algorithms, previously retrained with 100% of the data for inference on the blind test set.

## Technologies and Libraries

*   Python
*   Pandas
*   Scikit-Learn
*   Matplotlib and Seaborn

## Author

**David Riquelme Riquelme**
[GitHub Profile](https://github.com/RiquelmeRiquelme)