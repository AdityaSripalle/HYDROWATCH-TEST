import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Streamlit App Configuration
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define all models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=4, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(kernel='linear', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
            'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, objective='multi:softmax', num_class=3, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate training accuracy
            training_accuracy = accuracy_score(y_train, y_train_pred)
            testing_accuracy = accuracy_score(y_test, y_test_pred)
            
            results[name] = {
                'Training Accuracy': training_accuracy,
                'Testing Accuracy': testing_accuracy,
                'Precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'F1 Score': f1_score(y_test, y_test_pred, average='weighted'),
                'R2 Score': r2_score(y_test, y_test_pred),
                'MAE': mean_absolute_error(y_test, y_test_pred)
            }
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results).T
        st.subheader("📊 Model Performance Table")
        st.dataframe(results_df)
        
        # Visualization - Model Comparison
        st.subheader("📈 Model Performance Metrics")
        metrics = ['Precision', 'F1 Score', 'R2 Score', 'MAE']
        
        for metric in metrics:
            fig, ax = plt.subplots()
            sns.barplot(x=results_df.index, y=results_df[metric], palette='coolwarm', ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel(metric)
            plt.title(f"{metric} Comparison Across Models")
            for i, v in enumerate(results_df[metric]):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
            st.pyplot(fig)
        
        # Cross-Val Accuracy vs Testing Accuracy
        st.subheader("📈 Training Accuracy vs Testing Accuracy")
        fig, ax = plt.subplots()
        results_df[['Training Accuracy', 'Testing Accuracy']].plot(kind='bar', ax=ax, figsize=(10, 5), colormap='coolwarm')
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy vs Testing Accuracy for Different Models")
        plt.legend()
        st.pyplot(fig)
        
        # Prediction Interface
        st.subheader("🔮 Predict Water Quality")
        user_inputs = [st.number_input(f"{feature}", value=0.0) for feature in features]
        input_scaled = scaler.transform([user_inputs])
        
        if st.button("Predict"):
            best_model_name = max(results, key=lambda x: results[x]['Testing Accuracy'])
            best_model = models[best_model_name]
            prediction = best_model.predict(input_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🔍 Predicted Water Quality: {predicted_label}")
            
            wqi_value = user_inputs[-1]  # Assuming WQI is the last input field
            quality, drinking, irrigation = classify_water_quality(wqi_value)
            st.info(f"🌊 Water Quality: {quality}\n🥤 Drinking Suitability: {drinking}\n🌱 Irrigation Suitability: {irrigation}")

def classify_water_quality(wqi):
    if wqi <= 25:
        return "Very Poor", "Not Suitable for Drinking", "Not Suitable for Irrigation"
    elif 25 < wqi <= 50:
        return "Poor", "Not Suitable for Drinking", "Not Suitable for Irrigation"
    elif 50 < wqi <= 75:
        return "Fair", "Possibly Suitable for Drinking", "Possibly Suitable for Irrigation"
    elif 75 < wqi <= 90:
        return "Good", "Suitable for Drinking", "Suitable for Irrigation"
    else:
        return "Excellent", "Suitable for Drinking", "Suitable for Irrigation"
