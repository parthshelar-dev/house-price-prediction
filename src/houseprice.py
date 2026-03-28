import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).parent.parent

# Data load
df = pd.read_csv(BASE_DIR / "data/data.csv")
print(df.head())
print(df.info())
print(df.describe())

# Feature engineering
dt = datetime.now()
current_year = dt.year
df["HouseAge"] = current_year - df["YearBuilt"]
df = df.drop("YearBuilt", axis=1) 

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "HouseAge"
]

X = df[features]
y = df["SalePrice"]

sns.heatmap(df[features + ["SalePrice"]].corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Data cleaning
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# feature scaling (Z-score) for Linear & Ridge
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train_poly)
X_test_scale = scaler.transform(X_test_poly)

# Linear Regression
model_1 = LinearRegression()
model_1.fit(X_train_scale, y_train)  
y_pred_1 = model_1.predict(X_test_scale)
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)
print("Linear Regression MSE : ", mse_1)
print("Linear Regression R2 :", r2_1)

# Ridge Regression
model_2 = Ridge(alpha=0.1)
model_2.fit(X_train_scale, y_train)
y_pred_2 = model_2.predict(X_test_scale)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)
print("Ridge Regression MSE : ", mse_2)
print("Ridge Regression R2 :", r2_2)


# Random Forest Regression
model_3 = RandomForestRegressor(n_estimators=100, random_state=42)
model_3.fit(X_train, y_train)
y_pred_3 = model_3.predict(X_test)
mse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)
print("Random Forest MSE:", mse_3)
print("Random Forest R2 :", r2_3)

plt.figure(figsize=(15, 5))
# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2 )
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression : Actual vs predicted Prices")

# Ridge Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2 )
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Ridge Regression : Actual vs predicted Prices")

# Random Forest Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2 )
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Regression : Actual vs predicted Prices")

plt.tight_layout()
plt.show()

# Model Selection
models = {
    "Linear": model_1,
    "Ridge": model_2,
    "RandomForest": model_3
}

mse_scores = {
    "Linear": mse_1,
    "Ridge": mse_2,
    "RandomForest": mse_3
}

best_model_name = min(mse_scores, key=mse_scores.get)
best_model = models[best_model_name]
print(f"Best Model : {best_model_name}")

joblib.dump(best_model, BASE_DIR / "model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
joblib.dump(poly, BASE_DIR / "poly.pkl")
joblib.dump(best_model_name, BASE_DIR / "best_model_name.pkl")

print("Model saved successfully")
