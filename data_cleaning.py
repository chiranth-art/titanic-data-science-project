
# ==================================================
# TITANIC PROJECT - STAGE 2: DATA CLEANING
# ==================================================

# Step 1: Import libraries
import ssl
import certifi
import seaborn as sns
import pandas as pd


# Step 2: Configure SSL certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context


# Step 3: Load original Titanic dataset
print("\n========== LOADING DATASET ==========")

df = sns.load_dataset("titanic")

print("Original Dataset Shape:", df.shape)


# Step 4: Check missing values
print("\n========== MISSING VALUES BEFORE CLEANING ==========")

print(df.isnull().sum())


# Step 5: Check duplicate records
print("\n========== DUPLICATE ANALYSIS ==========")

duplicate_count = df.duplicated().sum()

print("Duplicate-looking rows:", duplicate_count)

print("Note: We are preserving these rows because")
print("identical attributes do not prove duplicate passengers.")


# Step 6: Create a working copy
cleaned_df = df.copy()


# Step 7: Remove unnecessary and redundant columns
print("\n========== REMOVING UNNECESSARY COLUMNS ==========")

columns_to_drop = [
    "deck",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alive",
    "alone"
]

cleaned_df = cleaned_df.drop(columns=columns_to_drop)

print("Removed columns:", columns_to_drop)


# Step 8: Handle missing age values using median
print("\n========== HANDLING MISSING AGE ==========")

age_median = cleaned_df["age"].median()

cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

print("Age missing values filled with median:", age_median)


# Step 9: Handle missing embarked values using mode
print("\n========== HANDLING MISSING EMBARKED VALUES ==========")

embarked_mode = cleaned_df["embarked"].mode()[0]

cleaned_df["embarked"] = cleaned_df["embarked"].fillna(
    embarked_mode
)

print("Embarked missing values filled with:", embarked_mode)


# Step 10: Detect outliers using IQR method
print("\n========== OUTLIER DETECTION ==========")

def detect_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    print("\nColumn:", column)
    print("Lower Limit:", round(lower_limit, 2))
    print("Upper Limit:", round(upper_limit, 2))
    print("Outliers Found:", len(outliers))


detect_outliers(cleaned_df, "age")
detect_outliers(cleaned_df, "fare")

print("\nOutliers are detected but not deleted.")
print("Some unusual fares and ages may be genuine.")


# Step 11: Feature engineering
print("\n========== FEATURE ENGINEERING ==========")

cleaned_df["family_size"] = (
    cleaned_df["sibsp"] + cleaned_df["parch"] + 1
)

cleaned_df["is_alone"] = (
    cleaned_df["family_size"] == 1
).astype(int)

print("Created family_size column.")
print("Created is_alone column.")


# Step 12: Convert gender into numerical values
print("\n========== ENCODING GENDER ==========")

cleaned_df["sex"] = cleaned_df["sex"].map({
    "female": 0,
    "male": 1
})

print("Female = 0")
print("Male = 1")


# Step 13: One-hot encode embarked column
print("\n========== ENCODING EMBARKED ==========")

cleaned_df = pd.get_dummies(
    cleaned_df,
    columns=["embarked"],
    drop_first=True,
    dtype=int
)

print("Categorical encoding completed.")


# Step 14: Check missing values after cleaning
print("\n========== MISSING VALUES AFTER CLEANING ==========")

print(cleaned_df.isnull().sum())


# Step 15: Validate cleaned dataset
print("\n========== CLEANED DATASET INFORMATION ==========")

print("Dataset Shape:", cleaned_df.shape)

print("\nFirst 5 rows:")
print(cleaned_df.head())

print("\nData types:")
print(cleaned_df.dtypes)

assert cleaned_df.isnull().sum().sum() == 0, \
    "Missing values still exist!"

print("\nValidation successful: No missing values found.")


# Step 16: Export cleaned dataset
print("\n========== EXPORTING CLEANED DATA ==========")

cleaned_df.to_csv("cleaned_titanic.csv", index=False)

print("File saved successfully: cleaned_titanic.csv")


# Step 17: Final summary
print("\n========== FINAL SUMMARY ==========")

print("Original rows:", len(df))
print("Cleaned rows:", len(cleaned_df))
print("Original columns:", df.shape[1])
print("Cleaned columns:", cleaned_df.shape[1])
print("Remaining missing values:", cleaned_df.isnull().sum().sum())

print("\nSTAGE 2: DATA CLEANING COMPLETED SUCCESSFULLY!")
