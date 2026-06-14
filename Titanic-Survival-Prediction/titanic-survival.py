# ==========================================
# TITANIC SURVIVAL PREDICTION
# CODSOFT TASK 1
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix

)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Titanic-Dataset.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# ==========================================
# DATASET INFO
# ==========================================

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== DATASET INFO ==========\n")
df.info()

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ==========================================
# SURVIVAL ANALYSIS
# ==========================================

print("\n========== SURVIVAL COUNT ==========\n")
print(df["Survived"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Distribution")
plt.savefig("survival_distribution.png")
plt.show()

# ==========================================
# GENDER ANALYSIS
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.savefig("gender_analysis.png")
plt.show()

# ==========================================
# PASSENGER CLASS ANALYSIS
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("class_analysis.png")
plt.show()

# ==========================================
# AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# ==========================================
# AGE VS SURVIVAL
# ==========================================

plt.figure(figsize=(8,5))
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.savefig("age_vs_survival.png")
plt.show()

# ==========================================
# FARE VS SURVIVAL
# ==========================================

plt.figure(figsize=(8,5))
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.savefig("fare_vs_survival.png")
plt.show()

# ==========================================
# DATA CLEANING
# ==========================================

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Remove Cabin column
df = df.drop(columns=["Cabin"])

# ==========================================
# ENCODING
# ==========================================

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Convert Embarked into numbers
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)

# ==========================================
# REMOVE UNUSED COLUMNS
# ==========================================

df = df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket"
    ]
)

# ==========================================
# REMOVE ANY LEFTOVER NaN
# ==========================================

df = df.dropna()

print("\n========== FINAL NULL VALUES ==========\n")
print(df.isnull().sum())

# ==========================================
# CONVERT ALL TO NUMERIC
# ==========================================

for col in df.columns:
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)
        # ==========================================
# FEATURE ENGINEERING
# ==========================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)

print("\n========== FEATURE ENGINEERING ==========\n")
print(df[["FamilySize", "IsAlone"]].head())

# ==========================================
# HEATMAP
# ==========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\n========== X SHAPE ==========\n")
print(X.shape)
# ==========================================
# FAMILY SIZE ANALYSIS
# ==========================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="FamilySize",
    hue="Survived",
    data=df
)

plt.title("Family Size vs Survival")
plt.savefig("family_size_analysis.png")
plt.show()

# ==========================================
# ALONE PASSENGER ANALYSIS
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="IsAlone",
    hue="Survived",
    data=df
)

plt.title("Alone vs Survival")
plt.savefig("alone_analysis.png")
plt.show()

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)
# ==========================================
# RANDOM FOREST MODEL
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\n========== RANDOM FOREST ACCURACY ==========\n")
print(f"{rf_accuracy * 100:.2f}%")

print("\n========== ACCURACY ==========\n")
print(f"{accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========\n")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\n========== CONFUSION MATRIX ==========\n")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# FINAL CONCLUSION
# ==========================================

print("""
PROJECT INSIGHTS

1. Female passengers had significantly higher survival chances.

2. First-class passengers survived more frequently than third-class passengers.

3. Fare was positively associated with survival.

4. Passengers travelling alone showed different survival patterns compared to families.

5. Family Size provided additional information for prediction.

6. Logistic Regression achieved approximately 81% accuracy.

7. Random Forest was used for comparison and feature importance analysis.

8. Sex, Fare and Passenger Class were among the most influential factors.
""")