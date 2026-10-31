
import ssl
import certifi
import seaborn as sns
import matplotlib.pyplot as plt

# SSL configuration
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Set graph style
sns.set_theme(style="whitegrid")

# 1. Survival count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("survival_count.png")
plt.close()

# 2. Survival by gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_gender.png")
plt.close()

# 3. Survival by passenger class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_class.png")
plt.close()

# 4. Age distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()

# 5. Survival by gender and class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("gender_class.png")
plt.close()

print("\nAll five visualizations created successfully!")

print("\nGenerated files:")
print("1. survival_count.png")
print("2. survival_gender.png")
print("3. survival_class.png")
print("4. age_distribution.png")
print("5. gender_class.png")
