# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using multiple machine learning models.
It includes data preprocessing, feature engineering, model training, evaluation, and prediction.

The goal is to compare different models and select the best one based on performance.

---

## 🚀 Models Used

* Linear Regression
* Ridge Regression (Best Model ✅)
* Random Forest Regression

---

## 📊 Features Used

* Area (sqft)
* Bedrooms
* Bathrooms
* Floors
* House Age (Derived from Year Built)

---

## ⚙️ Techniques Applied

* Feature Engineering (House Age creation)
* Polynomial Features (degree = 2)
* Feature Scaling (StandardScaler)
* One-Hot Encoding (pd.get_dummies)
* Model Evaluation using MSE (Mean Squared Error)
* Heatmap for Feature Correlation

---

## 📈 Model Performance

| Model             | MSE            |
| ----------------- | -------------- |
| Linear Regression | (your value)   |
| Ridge Regression  | (your value) ✅ |
| Random Forest     | (your value)   |

👉 Ridge Regression performed best with the lowest MSE.

---

## 📊 Visualizations

### 🔹 Correlation Heatmap

(Insert heatmap screenshot here)

### 🔹 Actual vs Predicted Graph

(Insert model comparison graph here)

---

## 🧠 Key Insights

* Polynomial features increased model complexity
* Ridge Regression reduced overfitting using regularization
* Random Forest did not perform best on this dataset
* Proper feature engineering improved accuracy

---

## ▶️ How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Run prediction

```bash
python src/predict.py
```

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── data.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model.pkl
├── scaler.pkl
├── poly.pkl
├── README.md
└── requirements.txt
```

---

## 📷 Sample Output

```
🏠 Enter Your House Details:
Area: 2000
Bedrooms: 3
Bathrooms: 2
Floors: 1
Year Built: 2015

🏠 Estimated Price: ₹ 82,45,000
```

---

## 👨‍💻 Author

Parth Shelar

---

## ⭐ Future Improvements

* Deploy as a web app (Streamlit)
* Add more features for better accuracy
* Hyperparameter tuning
* Use advanced models (XGBoost, etc.)
