import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE

# Streamlit App Theme Configuration
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")
st.title("💧 Water Quality Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

def load_data(uploaded_file):
    """Loads and preprocesses the dataset."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS', 'WQI']
        target = 'Water Quality Classification'

        if target not in df.columns:
            st.error("❌ Error: The dataset does not contain the expected target column.")
            return None, None, None, None

        df_cleaned = df[features + [target]].copy()
        df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)

        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    else:
        return None, None, None, None

# Display Data Overview
st.subheader("📊 Data Review")
if uploaded_file:
    df_cleaned, features, target, label_encoder = load_data(uploaded_file)
    if df_cleaned is not None:
        st.write(df_cleaned.head())

        X = df_cleaned[features]
        y = df_cleaned[target]

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', C=1, random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)  # Predictions on training data
            y_test_pred = model.predict(X_test)  # Predictions on testing data

            results[name] = {
                'Training Accuracy': accuracy_score(y_train, y_train_pred),
                'Testing Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'F1 Score': f1_score(y_test, y_test_pred, average='weighted'),
                'R2 Score': r2_score(y_test, y_test_pred),
                'MAE': mean_absolute_error(y_test, y_test_pred)
            }

        # Convert results to DataFrame for display
        results_df = pd.DataFrame(results).T
        st.subheader("📊 Model Performance (Training vs Testing Accuracy)")
        st.dataframe(results_df)

        # Select best model based on Testing Accuracy
        best_model_name = max(results, key=lambda x: results[x]['Testing Accuracy'])
        best_model = models[best_model_name]

        # Prediction Interface
        st.subheader("🔮 Predict Water Quality")
        user_inputs = [st.number_input(f"{feature}", value=0.0) for feature in features]
        if st.button("Predict"):
            input_scaled = scaler.transform([user_inputs])
            prediction = best_model.predict(input_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🔍 Predicted Water Quality: {predicted_label}")
