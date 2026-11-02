
# ==========================================
# TITANIC DATA SCIENCE PROJECT
# STAGE 6.2 - MODEL PERFORMANCE DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Titanic Data Science Dashboard",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------
# LOAD MODELS AND DATA
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


@st.cache_data
def load_dataset():

    return pd.read_csv("cleaned_titanic.csv")


logistic_model, random_forest_model, scaler = load_models()
df = load_dataset()

# ------------------------------------------
# HEADER
# ------------------------------------------

st.title("🚢 Titanic Data Science Dashboard")

st.markdown(
    """
    Explore passenger survival predictions, compare machine learning
    models, and understand patterns in the Titanic dataset.
    """
)

st.sidebar.header("About This Project")

st.sidebar.write(
    """
    **Project:** Titanic Survival Prediction

    **Machine Learning Models:**
    - Logistic Regression
    - Random Forest

    **Dataset:** Titanic Passenger Dataset

    **Technologies:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Matplotlib
    """
)

st.sidebar.divider()
st.sidebar.caption("Titanic Data Science Project | Stage 6.2")

# ------------------------------------------
# DASHBOARD TABS
# ------------------------------------------

prediction_tab, performance_tab, insights_tab = st.tabs(
    [
        "👤 Survival Prediction",
        "📊 Model Performance",
        "🔎 Dataset Insights"
    ]
)

# ==========================================
# TAB 1 - SURVIVAL PREDICTION
# ==========================================

with prediction_tab:

    st.header("Passenger Survival Prediction")

    st.write(
        "Enter passenger information to predict survival "
        "using Logistic Regression and Random Forest."
    )

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

    if submitted:

        # Feature engineering
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Encoding
        sex_encoded = 0 if sex == "female" else 1
        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0

        # Prepare passenger data
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

        # Logistic Regression prediction
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

        # Random Forest prediction
        rf_prediction = random_forest_model.predict(
            passenger_data
        )[0]

        rf_probabilities = random_forest_model.predict_proba(
            passenger_data
        )[0]

        rf_survival_probability = rf_probabilities[
            list(random_forest_model.classes_).index(1)
        ]

        # Display results
        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Logistic Regression")

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

            st.markdown("#### Random Forest")

            if rf_prediction == 1:
                st.success("Prediction: SURVIVED")
            else:
                st.error("Prediction: DID NOT SURVIVE")

            st.metric(
                "Survival Probability",
                f"{rf_survival_probability * 100:.2f}%"
            )

            st.progress(float(rf_survival_probability))

        # Model comparison
        st.divider()
        st.subheader("Prediction Comparison")

        comparison = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest"
            ],
            "Prediction": [
                "Survived" if logistic_prediction == 1
                else "Did Not Survive",
                "Survived" if rf_prediction == 1
                else "Did Not Survive"
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
                st.success(
                    "Both models predict that the passenger survived."
                )
            else:
                st.warning(
                    "Both models predict that the passenger "
                    "did not survive."
                )

        else:
            st.warning(
                "The models disagree. Consider both predictions carefully."
            )

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
            "These are machine learning estimates, "
            "not guaranteed historical outcomes."
        )

# ==========================================
# TAB 2 - MODEL PERFORMANCE
# ==========================================

with performance_tab:

    st.header("Machine Learning Model Evaluation")

    st.write(
        """
        Both models are evaluated on the same held-out test dataset.
        The test split uses 80% training data and 20% testing data,
        with random_state=42 and stratification.
        """
    )

    # Recreate the same test split used during training
    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Generate predictions
    X_test_scaled = scaler.transform(X_test)

    logistic_predictions = logistic_model.predict(
        X_test_scaled
    )

    rf_predictions = random_forest_model.predict(
        X_test
    )

    # Calculate evaluation metrics
    model_results = {}

    predictions_by_model = {
        "Logistic Regression": logistic_predictions,
        "Random Forest": rf_predictions
    }

    for model_name, predictions in predictions_by_model.items():

        model_results[model_name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, predictions, zero_division=0
            )
        }

    # Metrics table
    metrics_df = pd.DataFrame(model_results).T
    metrics_percent = metrics_df * 100
    metrics_percent = metrics_percent.round(2)

    st.subheader("Evaluation Metrics")

    metric_col1, metric_col2 = st.columns(2)

    for column, model_name in zip(
        [metric_col1, metric_col2],
        ["Logistic Regression", "Random Forest"]
    ):

        with column:

            st.markdown(f"#### {model_name}")

            st.metric(
                "Accuracy",
                f"{model_results[model_name]['Accuracy'] * 100:.2f}%"
            )

            st.metric(
                "Precision",
                f"{model_results[model_name]['Precision'] * 100:.2f}%"
            )

            st.metric(
                "Recall",
                f"{model_results[model_name]['Recall'] * 100:.2f}%"
            )

            st.metric(
                "F1-Score",
                f"{model_results[model_name]['F1-Score'] * 100:.2f}%"
            )

    st.divider()

    st.subheader("Complete Model Comparison")

    st.dataframe(
        metrics_percent,
        use_container_width=True
    )

    st.bar_chart(metrics_percent)

    # Confusion matrices
    st.divider()
    st.subheader("Confusion Matrices")

    st.write(
        """
        A confusion matrix compares actual passenger outcomes
        with the predictions made by a model.
        """
    )

    cm_col1, cm_col2 = st.columns(2)

    for column, model_name, predictions in [
        (cm_col1, "Logistic Regression", logistic_predictions),
        (cm_col2, "Random Forest", rf_predictions)
    ]:

        with column:

            st.markdown(f"#### {model_name}")

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Did Not Survive", "Actual: Survived"],
                columns=["Predicted: Did Not Survive", "Predicted: Survived"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

    st.caption(
        "For the confusion matrices, 0 = Did Not Survive and "
        "1 = Survived. Precision, recall and F1-score use "
        "Survived (class 1) as the positive class."
    )

    # Feature importance
    st.divider()
    st.subheader("Random Forest Feature Importance")

    try:

        importance_df = pd.read_csv(
            "models/feature_importance.csv"
        )

        feature_column = next(
            (
                column for column in ["Feature", "feature"]
                if column in importance_df.columns
            ),
            importance_df.columns[0]
        )

        numeric_columns = importance_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                "No numeric importance column was found."
            )

        importance_column = numeric_columns[-1]

        importance_chart = (
            importance_df
            .set_index(feature_column)[importance_column]
            .sort_values(ascending=True)
        )

        st.bar_chart(importance_chart)

        st.write(
            "Higher values indicate that a feature contributed "
            "more to the Random Forest model's decision process."
        )

        st.dataframe(
            importance_df.sort_values(
                by=importance_column,
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:

        st.warning(
            f"Could not display feature importance: {error}"
        )

# ==========================================
# TAB 3 - DATASET INSIGHTS
# ==========================================

with insights_tab:

    st.header("Titanic Dataset Insights")

    st.write(
        "Explore the passenger data and discover patterns "
        "associated with survival."
    )

    total_passengers = len(df)
    survived_count = int(df["survived"].sum())
    not_survived_count = total_passengers - survived_count
    survival_rate = (
        survived_count / total_passengers * 100
    )

    # Dataset summary
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Passengers",
        f"{total_passengers:,}"
    )

    col2.metric(
        "Survived",
        f"{survived_count:,}"
    )

    col3.metric(
        "Did Not Survive",
        f"{not_survived_count:,}"
    )

    col4.metric(
        "Survival Rate",
        f"{survival_rate:.2f}%"
    )

    st.divider()

    # Overall survival
    st.subheader("Overall Survival Distribution")

    survival_distribution = pd.DataFrame({
        "Passenger Outcome": [
            "Survived",
            "Did Not Survive"
        ],
        "Passengers": [
            survived_count,
            not_survived_count
        ]
    })

    st.bar_chart(
        survival_distribution.set_index("Passenger Outcome")
    )

    # Survival by gender
    st.divider()
    st.subheader("Survival Rate by Gender")

    gender_data = df.copy()

    gender_data["Gender"] = gender_data["sex"].map({
        0: "Female",
        1: "Male"
    })

    gender_survival = (
        gender_data
        .groupby("Gender")["survived"]
        .mean()
        .mul(100)
        .reset_index(name="Survival Rate (%)")
    )

    st.bar_chart(
        gender_survival.set_index("Gender")
    )

    st.dataframe(
        gender_survival.round(2),
        use_container_width=True,
        hide_index=True
    )

    # Survival by passenger class
    st.divider()
    st.subheader("Survival Rate by Passenger Class")

    class_survival = (
        df.groupby("pclass")["survived"]
        .mean()
        .mul(100)
        .reset_index(name="Survival Rate (%)")
    )

    class_survival["Passenger Class"] = (
        class_survival["pclass"].map({
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class"
        })
    )

    st.bar_chart(
        class_survival.set_index("Passenger Class")[
            ["Survival Rate (%)"]
        ]
    )

    # Age distribution
    st.divider()
    st.subheader("Passenger Age Distribution")

    age_distribution = pd.cut(
        df["age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 100],
        include_lowest=True
    ).value_counts().sort_index()

    age_chart = age_distribution.rename_axis(
        "Age Group"
    ).to_frame("Passengers")

    age_chart.index = age_chart.index.astype(str)

    st.bar_chart(age_chart)

    # Fare distribution
    st.divider()
    st.subheader("Ticket Fare Distribution")

    fare_bins = pd.cut(
        df["fare"],
        bins=[-1, 10, 30, 50, 100, float("inf")],
        labels=[
            "0–10",
            "10–30",
            "30–50",
            "50–100",
            "Above 100"
        ]
    )

    fare_distribution = (
        fare_bins
        .value_counts()
        .sort_index()
        .rename_axis("Fare Range")
        .to_frame("Passengers")
    )

    st.bar_chart(fare_distribution)

    st.caption(
        "Dataset insights describe the cleaned Titanic dataset. "
        "They show historical patterns and do not prove causation."
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "Titanic Data Science Project | Stage 6.2 | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
