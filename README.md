# Heart Disease Prediction Project

## Overview

This project leverages machine learning to predict the likelihood of heart disease based on patient health data. It encompasses data preprocessing, feature engineering, dimensionality reduction, model training, and a user-friendly Streamlit application for real-time predictions.

## Key Features

### 1. Data Preprocessing
- **Missing Values**: Imputed using median for numerical features and mode for categorical features.
- **Encoding**: Applied Label Encoding to transform categorical variables.
- **Scaling**: Standardized numerical features using `StandardScaler`.
- **Outlier Detection**: Identified and analyzed outliers using the Interquartile Range (IQR) method.

### 2. Feature Selection
- **Feature Importance**: Ranked predictors using Random Forest feature importance.
- **Recursive Feature Elimination (RFE)**: Selected optimal features for model performance.
- **Statistical Testing**: Conducted Chi-Square tests to assess feature significance.
- **Reduced Dataset**: Created a streamlined dataset with the most predictive features.

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduced feature dimensionality while preserving 95% of variance.
- **Visualization**: Plotted cumulative explained variance and PCA-transformed data for insight.

### 4. Model Training
- **Algorithms**: Trained Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM) models.
- **Optimization**: Tuned hyperparameters using `GridSearchCV` and `RandomizedSearchCV`.
- **Model Persistence**: Saved the best-performing Random Forest model for deployment.

### 5. Streamlit Application
- **User Interface**: Built an intuitive UI for real-time heart disease predictions.
- **Functionality**: Allows users to input health metrics and view prediction probabilities.
- **Risk Assessment**: Displays risk levels based on prediction confidence.

## Project Structure

- **`data_preprocessing.ipynb`**: Notebook for data cleaning and preprocessing steps.
- **`feature_selection.ipynb`**: Notebook for feature selection and analysis.
- **`dimensionality_reduction_pca.ipynb`**: Notebook for PCA-based dimensionality reduction.
- **`classification_models.ipynb`**: Notebook for training and evaluating classification models.
- **`streamlit_ui.py`**: Streamlit script for the real-time prediction interface.
- **`trained_feature_names.txt`**: List of feature names used in the trained model.
- **`heart_disease_cleaned.csv`**: Preprocessed dataset.
- **`heart_disease_reduced_with_binary.csv`**: Reduced dataset with binary target variable.
- **`optimized_random_forest_model.pkl`**: Serialized Random Forest model for deployment.

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: Install via `pip install -r requirements.txt`.
- Ensure all datasets and the trained model file are in the project directory.

### Running the Streamlit App
1. Open a terminal in the project directory.
2. Run the following command:
   ```bash
   streamlit run streamlit_ui.py
   ```
3. Access the app via the local URL provided in the terminal (e.g., `http://localhost:8501`).

### Exploring Notebooks
- Use Jupyter Notebook to explore the preprocessing, feature selection, dimensionality reduction, and model training steps in the provided `.ipynb` files.

## Results
- The optimized Random Forest model delivers high accuracy and robust performance in predicting heart disease.
- The Streamlit app offers an accessible interface for users to input health data and receive instant predictions with risk levels.

## Acknowledgments
- **Dataset**: Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
- **Libraries**: Utilizes `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `Streamlit`.

## Deployment
The Streamlit application is live and accessible at:  
[Heart Disease Prediction App](https://heart-diseasepredict.streamlit.app/)
