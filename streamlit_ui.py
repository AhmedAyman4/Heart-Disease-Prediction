import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("optimized_random_forest_model.pkl")

# Streamlit UI
st.title("Heart Disease Prediction")
st.sidebar.header("User Input Features")

# Function to get user input
# Removed features not used by the model: 'sex', 'cp', 'fbs', 'restecg'
def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 246)
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    data = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
        "ca": ca,
        "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 3,
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Load trained feature names from file
with open('trained_feature_names.txt', 'r') as f:
    trained_feature_names = f.read().splitlines()

# Ensure input_df has the same feature names as the model was trained on
input_df = input_df.reindex(columns=trained_feature_names, fill_value=0)

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Heart Disease Present" if prediction[0] == 1 else "No Heart Disease")

st.subheader("Prediction Probability")
st.write(prediction_proba)

# Display prediction message based on probability
st.write(f"The prediction is based on a probability of {prediction_proba[0][1]:.2f} for heart disease presence.")

# Additional prediction message based on probability
probability = prediction_proba[0][1]

if probability >= 0.7:
    st.error(f"⚠️ High risk: This person would likely have heart disease (probability: {probability:.2f})")
elif probability >= 0.5:
    st.warning(f"⚠️ Moderate risk: This person may have heart disease (probability: {probability:.2f})")
elif probability >= 0.3:
    st.info(f"⚠️ Low-moderate risk: This person has some risk of heart disease (probability: {probability:.2f})")
else:
    st.success(f"✅ Low risk: This person would likely not have heart disease (probability: {probability:.2f})")


# st.subheader("Heart Disease Trends")

# # Load dataset for visualization
# data = pd.read_csv("heart_disease_cleaned.csv")

# # Correlation heatmap
# st.write("Correlation Heatmap")
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
# st.pyplot(fig)

# # Age distribution by heart disease presence
# st.write("Age Distribution by Heart Disease Presence")
# fig, ax = plt.subplots()
# sns.boxplot(data=data, x="num", y="age", palette="Set2", ax=ax)
# ax.set_title("Age Distribution by Heart Disease Presence")
# ax.set_xlabel("Heart Disease (0=No, 1=Yes)")
# ax.set_ylabel("Age")
# st.pyplot(fig)
