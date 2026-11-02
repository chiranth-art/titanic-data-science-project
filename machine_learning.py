
# ==========================================
# TITANIC DATA SCIENCE PROJECT
# STAGE 4 - MACHINE LEARNING
# ==========================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== STAGE 4: MACHINE LEARNING ==========\n")

# 1. Load cleaned dataset
print("Loading cleaned Titanic dataset...")

df = pd.read_csv("cleaned_titanic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate features and target
print("\nSelecting features and target...")

X = df.drop("survived", axis=1)
y = df["survived"]

print("Features:", X.columns.tolist())
print("Target: survived")

# 3. Split dataset into training and testing
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Feature scaling for Logistic Regression
print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")

# 5. Train Logistic Regression
print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("Logistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")

# 6. Train Random Forest
print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# 7. Compare both models
print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy (%)": [
        round(logistic_accuracy * 100, 2),
        round(rf_accuracy * 100, 2)
    ]
})

print(comparison.to_string(index=False))

# 8. Detailed evaluation
print("\n========== LOGISTIC REGRESSION REPORT ==========")

print(classification_report(y_test, logistic_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\n========== RANDOM FOREST REPORT ==========")

print(classification_report(y_test, rf_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# 9. Identify best model
if logistic_accuracy > rf_accuracy:
    best_model = "Logistic Regression"
elif rf_accuracy > logistic_accuracy:
    best_model = "Random Forest"
else:
    best_model = "Both models have equal accuracy"

print("\nBest model based on test accuracy:", best_model)

# 10. Save model comparison
comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved to model_comparison.csv")

print("\n========== STAGE 4 COMPLETED ==========")
# ==========================================
# 11. SAVE TRAINED MODELS
# ==========================================

print("\n========== SAVING TRAINED MODELS ==========")

os.makedirs("models", exist_ok=True)

# Save Logistic Regression
joblib.dump(
    logistic_model,
    "models/logistic_regression.pkl"
)

# Save Random Forest
joblib.dump(
    random_forest_model,
    "models/random_forest.pkl"
)

# Save feature scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save Random Forest feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("Logistic Regression model saved!")
print("Random Forest model saved!")
print("Feature scaler saved!")
print("Feature importance saved!")

print("\nAll models saved successfully!")