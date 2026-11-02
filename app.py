
# ==========================================
# TITANIC DATA SCIENCE PROJECT
# STAGE 5 - WEB APPLICATION
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------
# LOAD MODELS
# ------------------------------------------

@st.cache_resource
def load_models():

    logistic_model = joblib.load(
        "models/logistic_regression.pkl"
    )

    random_forest_model = joblib.load(
        "models/random_forest.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return logistic_model, random_forest_model, scaler


logistic_model, random_forest_model, scaler = load_models()

# ------------------------------------------
# HEADER
# ------------------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.markdown(
    """
    ### Machine Learning Based Survival Prediction

    Enter passenger details below to predict survival
    using Logistic Regression and Random Forest.
    """
)

st.divider()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------

st.sidebar.header("About This Project")

st.sidebar.write(
    """
    **Project:** Titanic Survival Prediction

    **Models Used:**
    - Logistic Regression
    - Random Forest

    **Dataset:** Titanic Passenger Dataset

    **Technology:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """
)

# ------------------------------------------
# PASSENGER INPUT FORM
# ------------------------------------------

st.header("👤 Passenger Information")

with st.form("passenger_form"):

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            format_func=lambda x: {
                1: "1st Class",
                2: "2nd Class",
                3: "3rd Class"
            }[x]
        )

        sex = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            value=80.0,
            step=1.0
        )

    with col2:

        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            ["C", "Q", "S"],
            format_func=lambda x: {
                "C": "Cherbourg",
                "Q": "Queenstown",
                "S": "Southampton"
            }[x]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Survival",
        use_container_width=True
    )

# ------------------------------------------
# PREDICTION
# ------------------------------------------

if submitted:

    # Feature engineering

    family_size = sibsp + parch + 1

    is_alone = 1 if family_size == 1 else 0

    # Encoding

    sex_encoded = 0 if sex == "female" else 1

    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    # Prepare input

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

    # Logistic Regression

    scaled_data = scaler.transform(passenger_data)

    logistic_prediction = logistic_model.predict(
        scaled_data
    )[0]

    logistic_probabilities = logistic_model.predict_proba(
        scaled_data
    )[0]

    logistic_survival_probability = logistic_probabilities[
        list(logistic_model.classes_).index(1)
    ]

    # Random Forest

    rf_prediction = random_forest_model.predict(
        passenger_data
    )[0]

    rf_probabilities = random_forest_model.predict_proba(
        passenger_data
    )[0]

    rf_survival_probability = rf_probabilities[
        list(random_forest_model.classes_).index(1)
    ]

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    st.divider()

    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Logistic Regression")

        if logistic_prediction == 1:
            st.success("Prediction: SURVIVED")
        else:
            st.error("Prediction: DID NOT SURVIVE")

        st.metric(
            "Survival Probability",
            f"{logistic_survival_probability * 100:.2f}%"
        )

        st.progress(float(logistic_survival_probability))

    with col2:

        st.subheader("Random Forest")

        if rf_prediction == 1:
            st.success("Prediction: SURVIVED")
        else:
            st.error("Prediction: DID NOT SURVIVE")

        st.metric(
            "Survival Probability",
            f"{rf_survival_probability * 100:.2f}%"
        )

        st.progress(float(rf_survival_probability))

    # --------------------------------------
    # MODEL COMPARISON
    # --------------------------------------

    st.divider()

    st.header("⚖️ Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Prediction": [
            "Survived" if logistic_prediction == 1 else "Did Not Survive",
            "Survived" if rf_prediction == 1 else "Did Not Survive"
        ],
        "Survival Probability (%)": [
            round(logistic_survival_probability * 100, 2),
            round(rf_survival_probability * 100, 2)
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    if logistic_prediction == rf_prediction:

        if logistic_prediction == 1:
            st.success("Both models predict that the passenger survived.")
        else:
            st.warning("Both models predict that the passenger did not survive.")

    else:
        st.warning(
            "The models disagree. Consider both predictions carefully."
        )

    # --------------------------------------
    # PASSENGER SUMMARY
    # --------------------------------------

    with st.expander("View Passenger Details"):

        st.write({
            "Passenger Class": pclass,
            "Gender": sex,
            "Age": age,
            "Ticket Fare": fare,
            "Family Size": family_size,
            "Travelling Alone": bool(is_alone),
            "Embarkation": embarked
        })

    st.caption(
        "Note: These are machine learning estimates, "
        "not guaranteed historical outcomes."
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "Titanic Data Science Project | Stage 5 | "
    "Built with Python, Scikit-learn and Streamlit"
)
