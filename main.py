import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE

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

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Remove highly correlated features (above 0.9 correlation)
        corr_matrix = pd.DataFrame(X_resampled).corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
        X_resampled = pd.DataFrame(X_resampled).drop(columns=to_drop, axis=1)

        # Stratified Train-Test Split (70% Train, 30% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
        )

        # Scale data (After Splitting to prevent data leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models with optimized hyperparameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=60, max_depth=7, min_samples_split=8, min_samples_leaf=4, random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=60, learning_rate=0.05, max_depth=3, min_samples_split=6, random_state=42
            ),
            'SVM': SVC(kernel='rbf', C=0.4, random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=6, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        }

        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            # Cross-validation on full dataset
            cross_val_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
            cross_val_acc = round(np.mean(cross_val_scores), 4)
            test_acc = round(accuracy_score(y_test, y_test_pred), 4)

            # Reduce Overfitting if Cross-Validation Accuracy > Testing Accuracy
            if cross_val_acc > test_acc + 0.03:  # If cross-val accuracy is significantly higher
                params = model.get_params()

                # Reduce complexity only if parameter exists in the model
                if 'max_depth' in params:
                    model.set_params(max_depth=max(params['max_depth'] - 1, 3))
                if 'n_estimators' in params:
                    model.set_params(n_estimators=max(params['n_estimators'] - 10, 50))
                if 'C' in params:  # For SVM
                    model.set_params(C=params['C'] * 0.8)

                model.fit(X_train, y_train)  # Retrain model with reduced complexity
                y_test_pred = model.predict(X_test)
                test_acc = round(accuracy_score(y_test, y_test_pred), 4)  # Update test accuracy

            results[name] = {
                'Cross-Val Accuracy': cross_val_acc,
                'Testing Accuracy': test_acc,
                'Precision': round(precision_score(y_test, y_test_pred, average='weighted', zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_test_pred, average='weighted'), 4),
                'R2 Score': round(r2_score(y_test, y_test_pred), 4),
                'MAE': round(mean_absolute_error(y_test, y_test_pred), 4)
            }

        # Convert results to DataFrame for display
        results_df = pd.DataFrame(results).T
        st.subheader("📊 Model Performance (Cross-Val vs Testing Accuracy)")
        st.dataframe(results_df)

        # Select best model based on Testing Accuracy
        best_model_name = max(results, key=lambda x: results[x]['Testing Accuracy'])
        best_model = models[best_model_name]

        # Visualization - Cross-Validation vs Testing Accuracy
        st.subheader("📈 Cross-Val vs Testing Accuracy Comparison")
        fig, ax = plt.subplots()
        results_df[['Cross-Val Accuracy', 'Testing Accuracy']].plot(kind='bar', ax=ax, figsize=(10, 5), colormap='coolwarm')
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy")
        plt.title("Cross-Val vs Testing Accuracy for Different Models")
        plt.legend()
        st.pyplot(fig)

        # Prediction Interface
        st.subheader("🔮 Predict Water Quality")
        user_inputs = [st.number_input(f"{feature}", value=0.0) for feature in features if feature not in to_drop]
        if st.button("Predict"):
            input_scaled = scaler.transform([user_inputs])
            prediction = best_model.predict(input_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🔍 Predicted Water Quality: {predicted_label}")
