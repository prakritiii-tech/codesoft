import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Advertising.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["Total_Advertising"] = (
    df["TV"] +
    df["Radio"] +
    df["Newspaper"]
)

df["TV_Radio_Ratio"] = (
    df["TV"] /
    (df["Radio"] + 1)
)

print("\n========== NEW FEATURES ==========\n")
print(
    df[
        [
            "Total_Advertising",
            "TV_Radio_Ratio"
        ]
    ].head()
)

# ==========================================
# SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Sales"],
    bins=20,
    kde=True
)

plt.title("Sales Distribution")

plt.savefig(
    "sales_distribution.png"
)

plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "correlation_heatmap.png"
)

plt.show()

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["Sales"]
)

y = df["Sales"]

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
# LINEAR REGRESSION
# ==========================================

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

lr_predictions = lr.predict(
    X_test
)

print(
    "\n========== LINEAR REGRESSION ==========\n"
)

print(
    "R2 Score:",
    round(
        r2_score(
            y_test,
            lr_predictions
        ),
        3
    )
)

# ==========================================
# RANDOM FOREST
# ==========================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_predictions = rf.predict(
    X_test
)

print(
    "\n========== RANDOM FOREST ==========\n"
)

print(
    "R2 Score:",
    round(
        r2_score(
            y_test,
            rf_predictions
        ),
        3
    )
)

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test,
            rf_predictions
        ),
        2
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                y_test,
                rf_predictions
            )
        ),
        2
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    "\n========== FEATURE IMPORTANCE ==========\n"
)

print(
    importance
)

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

sample = pd.DataFrame(
    {
        "TV": [150],
        "Radio": [30],
        "Newspaper": [20],
        "Total_Advertising": [200],
        "TV_Radio_Ratio": [150 / 31]
    }
)

predicted_sales = rf.predict(
    sample
)

print(
    "\n========== CUSTOM PREDICTION ==========\n"
)

print(
    "Predicted Sales:",
    round(
        predicted_sales[0],
        2
    )
)

# ==========================================
# CONCLUSION
# ==========================================

print("""

PROJECT INSIGHTS

1. TV advertising showed the strongest influence on sales.

2. Radio advertising also contributed significantly.

3. Newspaper advertising had relatively lower impact.

4. Feature engineering improved model understanding.

5. Random Forest outperformed Linear Regression.

6. Total advertising budget strongly correlated with sales.

CONCLUSION

Machine Learning can effectively forecast product sales using advertising expenditure data.
""")