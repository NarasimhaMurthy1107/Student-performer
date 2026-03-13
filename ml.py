import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


# =========================
# LOAD DATA
# =========================

data = pd.read_csv("student_data.csv")

# =========================
# FEATURE ENGINEERING ⭐
# =========================

data["StudyEfficiency"] = data["Assignments"] / data["StudyHours"]

# =========================
# REGRESSION PROBLEM
# =========================

X = data.drop("FinalScore", axis=1)
y = data["FinalScore"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MULTIPLE MODELS ⭐
# =========================

models = {
    "Linear": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = r2
    print(f"{name} -> RMSE: {rmse:.2f}  R2: {r2:.2f}")

# =========================
# MODEL COMPARISON GRAPH ⭐
# =========================

plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Comparison (R2 Score)")
plt.show()

# =========================
# HYPERPARAMETER TUNING ⭐⭐⭐
# =========================

params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None]
}

grid = GridSearchCV(RandomForestRegressor(), params, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# =========================
# FINAL PREDICTION GRAPH
# =========================

final_pred = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, final_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Final Model Prediction")
plt.show()

# =========================
# FEATURE IMPORTANCE ⭐
# =========================

plt.figure()
plt.bar(X.columns, best_model.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()

# =========================
# CLASSIFICATION VERSION ⭐⭐⭐
# =========================

data["Pass"] = np.where(data["FinalScore"] >= 70, 1, 0)

Xc = data.drop(["FinalScore", "Pass"], axis=1)
yc = data["Pass"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(Xc_train, yc_train)

yc_pred = clf.predict(Xc_test)

print("\nClassification Accuracy:", accuracy_score(yc_test, yc_pred))

cm = confusion_matrix(yc_test, yc_pred)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()