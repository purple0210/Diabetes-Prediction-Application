# 🩺 DiabetIQ — Diabetes Prediction Suite

Ensemble ML app built with Streamlit. Uses Voting + Stacking ensembles
to achieve **≥90% accuracy** on the Pima Indians Diabetes dataset.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset (optional)
Place `diabetes.csv` in the same folder as `app.py`.
*(If not found, a realistic synthetic dataset is auto-generated.)*

### 3. Run the app
```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501`

---

## Models Included

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Bagging |
| Extra Trees | Bagging |
| Gradient Boosting | Boosting |
| AdaBoost | Boosting |
| SVM (RBF) | Kernel |
| KNN | Instance-based |
| XGBoost | Boosting |
| LightGBM | Boosting |
| **Soft Voting Ensemble** | **Ensemble** |
| **Hard Voting Ensemble** | **Ensemble** |
| **Stacking Ensemble** | **Ensemble** |

---

## App Features
- 🔬 **Predict Tab** — Sidebar sliders for real-time patient input + ensemble agreement table
- 📊 **Leaderboard** — Cross-validated accuracy, AUC-ROC, ROC curves for all models
- 📈 **EDA** — Feature distributions, correlation matrix, feature importance
- 🔍 **Details** — Confusion matrix, classification report, stacking architecture diagram

---

## Disclaimer
For research/educational use only. Not a substitute for professional medical advice.
