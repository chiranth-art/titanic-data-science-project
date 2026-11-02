https://titanic-data-science-project-qjneoan4kzkrsxpa7lx6vr.streamlit.app/?utm_source=chatgpt.com

# 🚢 Titanic Survival Prediction – Data Science & Machine Learning

A complete end-to-end Data Science and Machine Learning project that explores the Titanic passenger dataset, performs data cleaning, visualizes survival patterns, trains machine learning models, and predicts passenger survival through an interactive web application.

## 📌 Project Overview

The objective of this project is to analyze the Titanic passenger dataset and build machine learning models to predict whether a passenger survived the Titanic disaster based on passenger information.

Two machine learning algorithms are implemented and compared:

- Logistic Regression
- Random Forest Classifier

A Streamlit web application allows users to enter passenger details and view predictions from both models.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Joblib | Saving and loading models |
| Streamlit | Interactive web application |
| Git & GitHub | Version control |

## 🚀 Project Development Stages

### Stage 1 – Exploratory Data Analysis (EDA)

- Loaded the Titanic dataset using Seaborn.
- Explored dataset structure and statistical information.
- Analyzed missing values and duplicate-looking records.
- Studied survival distribution by gender and passenger class.
- Created initial data visualizations.

**Files:** `explore.py`, `visualization.py`

### Stage 2 – Data Cleaning and Preprocessing

- Handled missing age values using median imputation.
- Handled missing embarkation values using mode imputation.
- Removed unnecessary and redundant columns.
- Encoded gender and embarkation categories.
- Created `family_size` and `is_alone` features.
- Exported the cleaned dataset.

**Output:** `cleaned_titanic.csv`

**File:** `data_cleaning.py`

### Stage 3 – Advanced Data Visualization

Created visualizations to understand passenger survival patterns:

- Age distribution analysis.
- Fare analysis.
- Family survival analysis.
- Correlation heatmap.
- Survival pattern analysis.

**Output:** `stage3_visualizations/`

**File:** `advanced_visualization.py`

### Stage 4 – Machine Learning

Trained and evaluated two classification algorithms.

**Logistic Regression**
- StandardScaler preprocessing.
- Model training and evaluation.
- Classification report and confusion matrix.

**Random Forest**
- Ensemble-based classification.
- Model training and evaluation.
- Feature importance analysis.

#### Model Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 80.45% |
| Random Forest | 81.01% |

Random Forest achieved the higher test accuracy in this experiment.

The dataset was divided into 80% training data and 20% testing data using a stratified split.

**Saved models:**
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/scaler.pkl`
- `models/feature_importance.csv`

**File:** `machine_learning.py`

### Stage 5 – Survival Prediction System

Developed a prediction system using the saved machine learning models.

Features:
- Passenger information input.
- Automatic feature engineering.
- Gender and embarkation encoding.
- Logistic Regression prediction.
- Random Forest prediction.
- Survival probability calculation.
- Model comparison.

**Python prediction program:** `predict_survival.py`

**Interactive web application:** `app.py`

## 📂 Project Structure

```text
titanic-data-science-project/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── feature_importance.csv
│
├── stage3_visualizations/
│   ├── age_distribution.png
│   ├── fare_analysis.png
│   ├── family_survival.png
│   ├── correlation_heatmap.png
│   └── survival_patterns.png
│
├── explore.py
├── visualization.py
├── data_cleaning.py
├── advanced_visualization.py
├── machine_learning.py
├── predict_survival.py
├── app.py
│
├── cleaned_titanic.csv
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chiranth-art/titanic-data-science-project.git
```

### 2. Navigate into the project

```bash
cd titanic-data-science-project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Running the Project

### Exploratory Data Analysis

```bash
python explore.py
```

### Initial Visualizations

```bash
python visualization.py
```

### Data Cleaning

```bash
python data_cleaning.py
```

### Advanced Visualizations

```bash
python advanced_visualization.py
```

### Machine Learning

```bash
python machine_learning.py
```

### Terminal-Based Prediction

```bash
python predict_survival.py
```

### Launch the Web Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

**Note:** The web application loads previously trained models. Run the machine learning training script first if the saved model files are unavailable.

## 📊 Prediction Input Features

The machine learning models use the following passenger features:

1. Passenger class
2. Gender
3. Age
4. Number of siblings and spouses
5. Number of parents and children
6. Ticket fare
7. Family size
8. Whether the passenger travelled alone
9. Embarkation port – Queenstown
10. Embarkation port – Southampton

## 🔮 Future Improvements

- Add prediction history.
- Add interactive model performance charts.
- Improve the web application interface.
- Deploy the application online.
- Explore additional machine learning algorithms.
- Add more detailed model evaluation and tuning.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes. Predictions are based on patterns learned from the Titanic dataset and should not be interpreted as guaranteed outcomes.

## 👨‍💻 Developer

**Chiranth**

GitHub: [chiranth-art](https://github.com/chiranth-art)

Project Repository: [Titanic Data Science Project](https://github.com/chiranth-art/titanic-data-science-project)

---

