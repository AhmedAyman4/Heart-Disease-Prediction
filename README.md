# Heart Disease Prediction Project

## Overview

This project aims to predict the likelihood of heart disease using machine learning models. The project includes data preprocessing, feature selection, dimensionality reduction, model training, and a user-friendly Streamlit application for real-time predictions.

## Key Features

1. **Data Preprocessing**:

   - Handled missing values using median and mode imputation.
   - Encoded categorical variables using Label Encoding.
   - Scaled numerical features using StandardScaler.
   - Detected and analyzed outliers using the IQR method.

2. **Feature Selection**:

   - Used Random Forest feature importance to rank predictors.
   - Applied Recursive Feature Elimination (RFE) to select the best features.
   - Performed Chi-Square tests to evaluate feature significance.
   - Created a reduced dataset with the most relevant features.

3. **Dimensionality Reduction**:

   - Applied Principal Component Analysis (PCA) to reduce feature dimensionality while retaining 95% variance.
   - Visualized cumulative explained variance and PCA-transformed data.

4. **Model Training**:

   - Trained multiple classification models, including Logistic Regression, Decision Tree, Random Forest, and SVM.
   - Optimized models using GridSearchCV and RandomizedSearchCV.
   - Saved the best-performing Random Forest model for deployment.

5. **Streamlit Application**:
   - Developed a user-friendly UI for real-time heart disease prediction.
   - Allowed users to input health data and view predictions with probabilities.
   - Displayed risk levels based on prediction probabilities.

## Files and Directories

- `data_preprocessing.ipynb`: Notebook for data cleaning and preprocessing.
- `feature_selection.ipynb`: Notebook for feature selection techniques.
- `dimensionality_reduction_pca.ipynb`: Notebook for PCA-based dimensionality reduction.
- `classification_models.ipynb`: Notebook for training and evaluating classification models.
- `streamlit_ui.py`: Streamlit application for real-time predictions.
- `trained_feature_names.txt`: File containing feature names used during model training.
- `heart_disease_cleaned.csv`: Cleaned dataset after preprocessing.
- `heart_disease_reduced_with_binary.csv`: Reduced dataset with binary target variable.
- `optimized_random_forest_model.pkl`: Trained Random Forest model saved for deployment.

## How to Run the Project

1. **Set Up Environment**:

   - Install required Python libraries: `pip install -r requirements.txt`.
   - Ensure all necessary files (e.g., datasets, model) are in the project directory.

2. **Run the Streamlit App**:

   - Execute the following command in the terminal:
     ```bash
     streamlit run streamlit_ui.py
     ```
   - Open the provided local URL in your browser to access the app.

3. **Explore Notebooks**:
   - Open the Jupyter notebooks to explore data preprocessing, feature selection, dimensionality reduction, and model training steps.

## Results

- The optimized Random Forest model achieved high accuracy and reliability in predicting heart disease.
- The Streamlit app provides an intuitive interface for users to input health data and receive predictions.

## Future Work

- Integrate additional machine learning models for comparison.
- Enhance the Streamlit app with more visualizations and insights.
- Deploy the app to a cloud platform for wider accessibility.

## Acknowledgments

- Dataset sourced from the UCI Machine Learning Repository.
- Libraries used: pandas, numpy, scikit-learn, matplotlib, seaborn, Streamlit.

## Deployment

The Streamlit application is deployed and accessible at the following link:
[Heart Disease Prediction App](https://heart-diseasepredict.streamlit.app/)
