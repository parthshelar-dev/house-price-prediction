import pandas as pd
import joblib
from datetime import datetime

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")
best_model_name = joblib.load("best_model_name.pkl")

print("Enter Your House Details:\n")

OverallQual = float(input("House Quality (1-10): "))
GrLivArea = float(input("Living Area (sq ft): "))
GarageCars = float(input("Garage Capacity (No. of cars): "))
TotalBsmtSF = float(input("Basement Area (sq ft): "))
FullBath = float(input("Number of Bathrooms: "))
YearBuilt = float(input("Year Built: "))

current_year = datetime.now().year
HouseAge = current_year - YearBuilt

user_data = pd.DataFrame([[
    OverallQual,
    GrLivArea,
    GarageCars,
    TotalBsmtSF,
    FullBath,
    HouseAge
]], columns=[
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "HouseAge"
])

if best_model_name == "RandomForest":
    price_predicted = model.predict(user_data)
else:
    user_poly = poly.transform(user_data)
    user_scaled = scaler.transform(user_poly)
    price_predicted = model.predict(user_scaled)

print(f"Predicted House Price: ${price_predicted[0]:,.2f}")