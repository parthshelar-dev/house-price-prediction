# 🏠 House Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts house prices using **Linear Regression**, **Ridge Regression**, and **Random Forest** — with feature engineering, polynomial features, and automatic model selection.

---

## 📌 Overview

This project builds a complete ML pipeline to predict house prices based on features like living area, garage capacity, basement area, bathrooms, overall quality, and house age.

It includes data preprocessing, feature engineering, model training, evaluation using MSE & R², and visualization — structured in a clean and beginner-friendly way.

---

## 🚀 Features

- ✅ Data loading and exploration
- ✅ Feature Engineering: `YearBuilt` → `HouseAge`
- ✅ Correlation Heatmap (EDA)
- ✅ Polynomial Features (degree = 2)
- ✅ Feature Scaling using `StandardScaler`
- ✅ Model training:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regression
- ✅ Model evaluation using MSE & R²
- ✅ Automatic best model selection
- ✅ Actual vs Predicted visualization
- ✅ Model saving using Joblib
- ✅ User input-based prediction system

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models & preprocessing |
| Matplotlib | Graph plotting |
| Seaborn | Heatmap visualization |
| Joblib | Model saving/loading |

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── data.csv
│
├── src/
│   ├── houseprice.py
│   └── predict.py
│
├── model.pkl
├── scaler.pkl
├── poly.pkl
├── best_model_name.pkl
│
├── images/
│   ├── heatmap.png
│   └── prediction_graph.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

```
Raw Data → Feature Engineering → Polynomial Features → Scaling → Train → Evaluate → Save → Predict
```

1. Load dataset from `data/data.csv`
2. Create new feature → `HouseAge` from `YearBuilt`
3. Apply Polynomial Features (degree = 2)
4. Scale features using `StandardScaler`
5. Train 3 models and evaluate using MSE & R²
6. Automatically select and save the best model
7. Run `predict.py` to enter house details and get a price

---

## 📊 Model Performance

| Model | MSE | R² |
|-------|-----|----|
| Linear Regression | 1.02e+09 | 0.841 |
| Ridge Regression | 1.04e+09 | 0.838 |
| Random Forest | 1.06e+09 | 0.836 |

Best Model = Linear Regression (lowest MSE)

---

## 📈 Visualizations

### 🔹 Correlation Heatmap
Shows the relationship between all features and Sale Price.

![Heatmap](images/heatmap.png)

### 🔹 Actual vs Predicted Prices
Points closer to the diagonal red line = better predictions.

![Prediction Graph](images/prediction_graph.png)

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Run prediction
```bash
python src/predict.py
```

---

## 💡 Key Insights

- `OverallQual` has the strongest correlation with `SalePrice` (~0.8)
- `HouseAge` shows a negative correlation — older houses are priced lower
- Polynomial features increased model complexity and improved Linear/Ridge performance
- Ridge Regression reduced overfitting using regularization
- Random Forest handles non-linearity well but needs more tuning for this dataset

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Try XGBoost / Gradient Boosting
- [ ] Add more features (neighborhood, roof style, etc.)
- [ ] Build a Streamlit web app
- [ ] Deploy online using Render or Hugging Face Spaces

---

## 👨‍💻 Author

**Parth Shelar**

---

⭐ If you found this useful, give it a star on GitHub!
