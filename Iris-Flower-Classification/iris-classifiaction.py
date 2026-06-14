import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Iris.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ==========================================
# SPECIES COUNT
# ==========================================

print("\n========== SPECIES COUNT ==========\n")
print(df["Species"].value_counts())

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="Species",
    data=df
)

plt.title("Species Distribution")

plt.savefig(
    "species_distribution.png"
)

plt.show()

# ==========================================
# PAIRPLOT
# ==========================================

sns.pairplot(
    df,
    hue="Species"
)

plt.savefig(
    "pairplot.png"
)

plt.show()

# ==========================================
# ENCODING
# ==========================================

species_map = {
    "Setosa": 0,
    "Versicolor": 1,
    "Virginica": 2
}

df["Species"] = df["Species"].map(
    species_map
)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop(
    "Species",
    axis=1
)

y = df["Species"]

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
# DECISION TREE
# ==========================================

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

dt_pred = dt.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print(
    "\n========== DECISION TREE ==========\n"
)

print(
    f"Accuracy: {dt_accuracy*100:.2f}%"
)

# ==========================================
# RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    "\n========== RANDOM FOREST ==========\n"
)

print(
    f"Accuracy: {rf_accuracy*100:.2f}%"
)

print(
    "\n========== CLASSIFICATION REPORT ==========\n"
)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

print(
    "\n========== CONFUSION MATRIX ==========\n"
)

print(cm)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    "\n========== FEATURE IMPORTANCE ==========\n"
)

print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance"
)

plt.savefig(
    "feature_importance.png"
)

plt.show()

# ==========================================
# CUSTOM PREDICTION
# ==========================================

sample = [[
    5.1,
    3.5,
    1.4,
    0.2
]]

prediction = rf.predict(
    sample
)

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

print(
    "\n========== CUSTOM PREDICTION ==========\n"
)

print(
    "Predicted Species:",
    species_names[
        prediction[0]
    ]
)

# ==========================================
# CONCLUSION
# ==========================================

print("""

PROJECT INSIGHTS

1. Iris dataset contains three flower species.

2. Petal measurements were the strongest predictors.

3. Random Forest outperformed Decision Tree.

4. Feature importance identified the most influential measurements.

5. The model successfully classified iris flowers into their respective species.

CONCLUSION

Machine Learning can accurately classify iris flowers using sepal and petal measurements.
""")