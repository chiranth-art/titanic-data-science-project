
# ==========================================
# TITANIC DATA SCIENCE PROJECT
# STAGE 5 - SURVIVAL PREDICTION SYSTEM
# ==========================================

import pandas as pd
import joblib

print("\n======================================")
print("     TITANIC SURVIVAL PREDICTOR")
print("======================================")

# 1. Load saved machine learning models
print("\nLoading trained models...")

logistic_model = joblib.load(
    "models/logistic_regression.pkl"
)

random_forest_model = joblib.load(
    "models/random_forest.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

print("All models loaded successfully!")

# 2. Collect passenger information
print("\n========== PASSENGER DETAILS ==========")

def get_integer(prompt, minimum, maximum=None):

    while True:
        try:
            value = int(input(prompt))

            if value < minimum:
                print("Value is too small. Try again.")
                continue

            if maximum is not None and value > maximum:
                print("Value is too large. Try again.")
                continue

            return value

        except ValueError:
            print("Please enter a valid whole number.")


def get_float(prompt, minimum):

    while True:
        try:
            value = float(input(prompt))

            if value < minimum:
                print("Value cannot be negative. Try again.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


pclass = get_integer(
    "Passenger class (1, 2, or 3): ", 1, 3
)

while True:
    sex = input("Gender (male/female): ").strip().lower()

    if sex in ["male", "female"]:
        break

    print("Please enter male or female.")

age = get_float("Age: ", 0)

sibsp = get_integer(
    "Number of siblings/spouses aboard: ", 0
)

parch = get_integer(
    "Number of parents/children aboard: ", 0
)

fare = get_float("Ticket fare: ", 0)

while True:
    embarked = input(
        "Port of embarkation (C/Q/S): "
    ).strip().upper()

    if embarked in ["C", "Q", "S"]:
        break

    print("Enter C, Q, or S.")

# 3. Feature engineering
family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0

# 4. Encode categorical variables
sex_encoded = 0 if sex == "female" else 1

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# 5. Create input dataframe
feature_columns = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "is_alone",
    "embarked_Q",
    "embarked_S"
]

passenger_data = pd.DataFrame(
    [[
        pclass,
        sex_encoded,
        age,
        sibsp,
        parch,
        fare,
        family_size,
        is_alone,
        embarked_Q,
        embarked_S
    ]],
    columns=feature_columns
)

print("\nPassenger information processed!")

# 6. Logistic Regression prediction
scaled_data = scaler.transform(passenger_data)

logistic_prediction = logistic_model.predict(
    scaled_data
)[0]

logistic_probability = logistic_model.predict_proba(
    scaled_data
)[0][list(logistic_model.classes_).index(1)]

# 7. Random Forest prediction
rf_prediction = random_forest_model.predict(
    passenger_data
)[0]

rf_probability = random_forest_model.predict_proba(
    passenger_data
)[0][list(random_forest_model.classes_).index(1)]

# 8. Display results
def display_result(model_name, prediction, probability):

    result = "SURVIVED" if prediction == 1 else "DID NOT SURVIVE"

    print(f"\n{model_name}")
    print("--------------------------------------")
    print("Prediction:", result)
    print("Survival Probability:", round(probability * 100, 2), "%")
    print("Non-survival Probability:", round((1 - probability) * 100, 2), "%")


print("\n======================================")
print("          PREDICTION RESULTS")
print("======================================")

display_result(
    "LOGISTIC REGRESSION",
    logistic_prediction,
    logistic_probability
)

display_result(
    "RANDOM FOREST",
    rf_prediction,
    rf_probability
)

print("\n======================================")
print("       STAGE 5 PREDICTION COMPLETE")
print("======================================")
