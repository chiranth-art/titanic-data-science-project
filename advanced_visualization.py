from matplotlib.patches import Rectangle
from typing import cast
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# STAGE 3: ADVANCED DATA VISUALIZATION
# ==========================================

print("\n========== STAGE 3: ADVANCED VISUALIZATION ==========\n")

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_titanic.csv")

# 2. Create output folder
output_folder = "stage3_visualizations"
os.makedirs(output_folder, exist_ok=True)

# 3. Configure visualization style
sns.set_theme(style="whitegrid", palette="deep")

# 4. Convert encoded values into readable labels
df["sex_label"] = df["sex"].map({
    0: "Female",
    1: "Male"
})

df["survival_label"] = df["survived"].map({
    0: "Did Not Survive",
    1: "Survived"
})

df["class_label"] = df["pclass"].map({
    1: "First Class",
    2: "Second Class",
    3: "Third Class"
})

print("Dataset loaded successfully!")
print("Total passengers:", len(df))
print("Total columns:", len(df.columns))


# ==========================================
# VISUALIZATION 1: AGE DISTRIBUTION
# ==========================================

print("\nGenerating Age Distribution...")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="age",
    hue="survival_label",
    bins=30,
    kde=True,
    alpha=0.45
)

plt.title("Age Distribution by Survival Status", fontsize=16)
plt.xlabel("Passenger Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(
    f"{output_folder}/age_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# VISUALIZATION 2: FARE ANALYSIS
# ==========================================

print("Generating Fare Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fare distribution
sns.histplot(
    data=df,
    x="fare",
    bins=40,
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Passenger Fare Distribution")
axes[0].set_xlabel("Ticket Fare")
axes[0].set_ylabel("Number of Passengers")

# Fare by passenger class
sns.boxplot(
    data=df,
    x="class_label",
    y="fare",
    showfliers=False,
    ax=axes[1]
)

axes[1].set_title("Ticket Fare by Passenger Class")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Ticket Fare")

plt.suptitle("Titanic Passenger Fare Analysis", fontsize=16)
plt.tight_layout()

plt.savefig(
    f"{output_folder}/fare_analysis.png",
    dpi=300
)

plt.close()


# ==========================================
# VISUALIZATION 3: FAMILY SIZE VS SURVIVAL
# ==========================================



# ==========================================
# VISUALIZATION 3: FAMILY SIZE VS SURVIVAL
# ==========================================

print("Generating Family Survival Analysis...")

# Calculate survival percentage for each family size
family_survival = (
    df.groupby("family_size")
    .agg(survival_rate=("survived", "mean"))
    .reset_index()
)

family_survival["survival_percentage"] = (
    family_survival["survival_rate"] * 100
)

# Create bar chart
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=family_survival,
    x="family_size",
    y="survival_percentage"
)

plt.title("Family Size vs Passenger Survival Rate", fontsize=16)
plt.xlabel("Family Size")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

# Display percentage labels above bars
# Display percentage labels above bars
for bar in ax.patches:
    rect = cast(Rectangle, bar)

    height = rect.get_height()

    ax.annotate(
        f"{height:.1f}%",
        (rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    f"{output_folder}/family_survival.png",
    dpi=300
)

plt.close()

print("Family Survival Analysis completed!")


numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Titanic Dataset Correlation Heatmap", fontsize=16)
plt.tight_layout()

plt.savefig(
    f"{output_folder}/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# VISUALIZATION 5: GENDER AND CLASS ANALYSIS
# ==========================================

print("Generating Gender and Class Survival Analysis...")

gender_class = (
    df.groupby(["sex_label", "class_label"])["survived"]
    .mean()
    .unstack()
)

gender_class = gender_class * 100

gender_class.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.7
)

plt.title("Survival Rate by Gender and Passenger Class", fontsize=16)
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.ylim(0, 110)
plt.legend(title="Passenger Class")
plt.tight_layout()

plt.savefig(
    f"{output_folder}/survival_patterns.png",
    dpi=300
)

plt.close()


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== STAGE 3 SUMMARY ==========")

print("Total Visualizations Generated: 5")

print("\nGenerated Files:")

for file in os.listdir(output_folder):
    if file.endswith(".png"):
        print(" -", file)

print("\nAll visualizations saved successfully!")
print("Output folder:", output_folder)

print("\n========== STAGE 3 COMPLETED SUCCESSFULLY ==========")
