
# ==========================================
# TITANIC DATASET - DATA EXPLORATION (EDA)
# ==========================================

# Step 1: Import libraries
import ssl
import certifi
import seaborn as sns
import pandas as pd

# Step 2: Configure SSL certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

# Step 3: Load Titanic dataset
df = sns.load_dataset("titanic")


# Step 4: Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# Step 5: Display last 5 rows
print("\n========== LAST 5 ROWS ==========")
print(df.tail())


# Step 6: Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()


# Step 7: Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# Step 8: Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())


# Step 9: Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# Step 10: Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate Rows:", df.duplicated().sum())


# Step 11: Basic statistical analysis
print("\n========== BASIC STATISTICS ==========")
print(df.describe())


# Step 12: Survival count
print("\n========== SURVIVAL COUNT ==========")
print(df["survived"].value_counts())

print("\n0 = Did Not Survive")
print("1 = Survived")


# Step 13: Survival percentage
print("\n========== SURVIVAL PERCENTAGE ==========")
print(df["survived"].value_counts(normalize=True) * 100)


# Step 14: Survival by gender
print("\n========== SURVIVAL BY GENDER ==========")
gender_survival = df.groupby("sex")["survived"].mean() * 100
print(gender_survival)


# Step 15: Survival by passenger class
print("\n========== SURVIVAL BY PASSENGER CLASS ==========")
class_survival = df.groupby("pclass")["survived"].mean() * 100
print(class_survival)


# Step 16: Average age of passengers
print("\n========== AVERAGE AGE ==========")
print("Average Age:", df["age"].mean())


# Step 17: Average ticket fare
print("\n========== AVERAGE TICKET FARE ==========")
print("Average Fare:", df["fare"].mean())


# Step 18: Survival by age group
print("\n========== SURVIVAL BY AGE GROUP ==========")

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)

age_survival = df.groupby(
    "age_group",
    observed=False
)["survived"].mean() * 100

print(age_survival)


# Step 19: Survival by gender and passenger class
print("\n========== GENDER AND CLASS ANALYSIS ==========")

print(
    df.pivot_table(
        values="survived",
        index="sex",
        columns="pclass",
        aggfunc="mean"
    ) * 100
)


# Step 20: Final summary
print("\n========== FINAL SUMMARY ==========")

print("Total Passengers:", len(df))
print("Survived:", (df["survived"] == 1).sum())
print("Did Not Survive:", (df["survived"] == 0).sum())
print("Average Passenger Age:", round(df["age"].mean(), 2))
print("Average Ticket Fare:", round(df["fare"].mean(), 2))

print("\n========== EDA COMPLETED SUCCESSFULLY ==========")
