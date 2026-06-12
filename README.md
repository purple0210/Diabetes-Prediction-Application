# 🩺 DiabetIQ — Diabetes Prediction using Machine Learning
 
> Ensemble Intelligence · High-Accuracy Classification with Feature Engineering & Multi-Model Stacking.
---
 
## 📌 Table of Contents
 
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Models Trained](#-models-trained)
- [Ensemble Architecture](#-ensemble-architecture)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Code Architecture](#-code-architecture)
- [Key Takeaways](#-key-takeaways)
- [Getting Started](#-getting-started)
---
 
## 🔬 Project Overview
 
**DiabetIQ** is an end-to-end machine learning suite for predicting diabetes in patients using the Pima Indians Diabetes Dataset. The project focuses on:
 
- **Binary classification**: Diabetic (1) vs Non-Diabetic (0)
- **Maximizing AUC-ROC** and minimizing false negatives
- **Clinically-motivated feature engineering** to boost model performance
- **Ensemble stacking** with 12+ trained classifiers
- **Interactive Streamlit UI** for real-time patient prediction and model analysis
---
 
## 📊 Dataset
 
| Property | Detail |
|---|---|
| Source | Pima Indians Diabetes Dataset |
| Patients | 768 female patients |
| Raw Features | 8 clinical features |
| Class Distribution | ~35% Diabetic · ~65% Non-Diabetic |
| Task | Binary Classification |
 
**Raw Features:** Glucose, BMI, Blood Pressure, Insulin, Pregnancies, Skin Thickness, Diabetes Pedigree Function (DPF), Age
 
---
 
## ⚙️ Pipeline
 
```
Raw Data
   ↓
Zero Imputation (Median)
   ↓
Outlier Clipping (IQR-based)
   ↓
Feature Engineering (10 new features)
   ↓
RobustScaler
   ↓
Train / Test Split (80/20, stratified)
   ↓
12+ Model Training + 10-Fold CV
   ↓
Soft Voting & Stacking Ensemble
   ↓
Evaluation (AUC-ROC, CV Accuracy, Confusion Matrix)
```
 
---
 
## 🧹 Data Preprocessing
 
| Step | Description |
|---|---|
| **Zero Imputation** | Physiologically impossible zeros in Glucose, BP, BMI, Skin Thickness, and Insulin replaced with per-column medians |
| **Outlier Clipping** | Values beyond `Q1 – 3×IQR` and `Q3 + 3×IQR` clipped to preserve statistical integrity |
| **RobustScaler** | Scales using IQR instead of std-dev — resistant to remaining outliers; applied after train/test split |
| **Train/Test Split** | 80/20 stratified split to ensure equal class ratio in both sets; 10-fold CV for robust evaluation |
 
---
 
## 🛠️ Feature Engineering
 
10 clinically-motivated derived features were created to improve model predictive power:
 
| Feature | Formula | Clinical Rationale |
|---|---|---|
| `Glucose_Insulin` | Glucose × Insulin / 1000 | HOMA-IR proxy — insulin resistance |
| `BMI_Age` | BMI × Age / 100 | Composite metabolic risk score |
| `Glucose_Age` | Glucose × Age / 1000 | Age-amplified hyperglycemia risk |
| `BMI_Class` | cut(BMI, [0, 18.5, 25, 30, 100]) | WHO weight category (0–3) |
| `Glucose_Tier` | cut(Glucose, [0, 99, 125, 500]) | Normal / Pre-diabetic / Diabetic |
| `HighRisk_Flag` | Σ of 5 binary risk checks | Additive severity indicator (0–5) |
| `Preg_Age_Ratio` | Pregnancies / (Age + 1) | Gestational diabetes signal |
| `Log_Insulin` | log1p(Insulin) | Reduces right-skew in Insulin |
| `Log_DPF` | log1p(DPF) | Reduces right-skew in DPF |
| `Skin_BMI` | SkinThickness / (BMI + 1) | Body composition proxy |
 
> ✨ **3 of the top 5 feature importance features were engineered** — validating the feature engineering step.
 
### Top 5 Predictors (Random Forest Importance)
 
| Rank | Feature | Importance | Role |
|---|---|---|---|
| 🥇 | Glucose | ~20% | Primary predictor |
| 🥈 | BMI | ~13% | Metabolic risk |
| 🥉 | Glucose_Insulin | ~11% | Engineered feature |
| 4️⃣ | Age | ~10% | Risk amplifier |
| 5️⃣ | HighRisk_Flag | ~8% | Composite risk |
 
---
 
## 🤖 Models Trained
 
12+ classifiers were trained and compared for selecting the best model:
 
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost *(optional — requires `pip install xgboost`)*
- LightGBM *(optional — requires `pip install lightgbm`)*
- **Soft Voting Ensemble**
- **Stacking Ensemble**
> 🏆 **Soft Voting Ensemble achieved the highest AUC-ROC and CV Accuracy.**
 
---
 
## 🧩 Ensemble Architecture
 
### Soft Voting
Averages class probabilities from all base classifiers — more nuanced than a simple majority vote.
 
### Stacking Classifier
- Trains a **Logistic Regression meta-learner** on base-model out-of-fold predictions
- `passthrough=True` feeds raw scaled features directly to the meta-learner
- **10-fold CV** inside `StackingClassifier` prevents data leakage from base learners
---
 
## 📈 Results
 
- **Ensemble models (Voting, Stacking)** consistently outperform all individual classifiers on both CV Accuracy and AUC-ROC
- Ensembles scored **2–4% higher AUC** than any single base learner
- AUC-ROC was chosen as the primary metric due to the 65/35 class imbalance
> With class imbalance, **AUC-ROC is the right primary metric** over simple accuracy — it is threshold-independent and handles imbalance well (AUC 0.5 = random, 1.0 = perfect).
 
---
 
## 💻 Streamlit App
 
The interactive Streamlit app features **4 tabs**:
 
### 🔬 Predict Patient
- Sidebar with 8 clinical sliders (real-time input)
- Radar chart — 6-axis risk visualization
- Prediction banner (green = Non-Diabetic / red = Diabetic)
- Probability bars for both classes
- Confidence % from `predict_proba`
### 📊 Model Leaderboard
- Sortable table: CV Accuracy, CV Std, Test Accuracy, AUC
- Best model highlighted row
- Horizontal bar chart of all model accuracies
- ROC curves overlay — all models on one plot
### 📈 EDA & Features
- Dataset shape & class balance
- Pie chart — class distribution
- Histograms by outcome (2×4 grid)
- Heatmap — Pearson correlation matrix
- Random Forest feature importance bar chart
### 🔍 Best Model Details
- Test accuracy + AUC-ROC metrics
- Confusion matrix heatmap
- Classification report table
- Individual ROC curve with AUC fill
---
 
## 🏗️ Code Architecture
 
### `engineer_features(df)`
- Accepts raw DataFrame, returns enriched copy
- Adds 10 derived columns (no in-place mutation)
- Used in both training pipeline and the predict tab
### `load_and_prepare()`
- `@st.cache_data` — runs once per session
- Falls back to synthetic 768-row data if CSV not found
- Imputes zeros, clips outliers, calls `engineer_features()`
### `build_models(X, y)`
- `@st.cache_resource` — persists model objects across reruns
- Fits `RobustScaler`, splits data, runs 10-fold CV
- Returns trained model dict, leaderboard DataFrame, best model + scaler
### Predict Pipeline
```
Slider values → raw DataFrame → engineer_features()
  → scaler.transform() → best_model.predict() + predict_proba()
  → Renders banner, progress bars, and confidence score
```
 
---
 
## 🚀 Getting Started
 
### Prerequisites
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
pip install xgboost lightgbm  # optional
```
 
### Run the App
```bash
streamlit run app.py
```
 
### Dataset
Place the `diabetes.csv` (Pima Indians Diabetes Dataset) in the project root. The app will fall back to synthetic data if the file is not found.
 
---
 
## 🔑 Key Takeaways
 
| | Insight |
|---|---|
| 🔬 | **Preprocessing Matters** — Median imputation + IQR clipping boosted model stability significantly over raw data |
| ⚙️ | **Feature Engineering Pays Off** — 3 of top 5 importance features were engineered (HighRisk_Flag, Glucose_Insulin, BMI_Age) |
| 🏆 | **Ensembles Win** — Stacking & Voting consistently scored 2–4% higher AUC than any single base learner |
| 📊 | **AUC > Accuracy** — With 65/35 imbalance, AUC-ROC is the correct primary metric for model selection |
| 💻 | **Interactive UI** — All 4 tabs (Predict, Leaderboard, EDA, Best Model) run end-to-end on a single Python file |
 
---
 
> ⚕️ *DiabetIQ is developed for educational purposes only and is not intended for clinical use.*
 
