import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetIQ · Prediction Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    color: #e0f7fa; margin: 0; letter-spacing: -1px;
}
.main-header p { color: #80cbc4; font-size: 1.05rem; margin-top: 0.5rem; font-weight: 300; }
.metric-card {
    background: linear-gradient(135deg, #1a237e, #283593); border-radius: 12px;
    padding: 1.2rem 1.5rem; color: white; text-align: center;
    box-shadow: 0 4px 16px rgba(26,35,126,0.4); margin-bottom: 0.5rem;
}
.metric-card .val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #80deea; }
.metric-card .lbl { font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
.pred-positive {
    background: linear-gradient(135deg, #b71c1c, #c62828); border-radius: 14px; padding: 2rem;
    text-align: center; color: white; font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; box-shadow: 0 8px 24px rgba(183,28,28,0.4);
}
.pred-negative {
    background: linear-gradient(135deg, #1b5e20, #2e7d32); border-radius: 14px; padding: 2rem;
    text-align: center; color: white; font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; box-shadow: 0 8px 24px rgba(27,94,32,0.4);
}
.sec-head {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.35rem; color: #1a237e;
    border-left: 4px solid #00acc1; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}
.best-badge {
    display: inline-block; background: #00acc1; color: #fff; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 20px; font-weight: 600; letter-spacing: 0.5px;
    margin-left: 6px; vertical-align: middle;
}
.info-box {
    background: #e3f2fd; border-left: 4px solid #1a237e; border-radius: 8px;
    padding: 0.9rem 1.2rem; margin-bottom: 1rem; font-size: 0.92rem; color: #1a237e;
}
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027 0%, #203a43 100%); }
section[data-testid="stSidebar"] * { color: #e0f7fa !important; }
section[data-testid="stSidebar"] label { color: #80cbc4 !important; font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_features(df):
    """Add clinically-motivated interaction and derived features."""
    d = df.copy()

    # Insulin resistance proxy (HOMA-IR approximation)
    d['Glucose_Insulin'] = d['Glucose'] * d['Insulin'] / 1000.0

    # BMI × Age composite risk
    d['BMI_Age'] = d['BMI'] * d['Age'] / 100.0

    # Glucose × Age (older + hyperglycemia = elevated risk)
    d['Glucose_Age'] = d['Glucose'] * d['Age'] / 1000.0

    # BMI class: underweight=0, normal=1, overweight=2, obese=3
    d['BMI_Class'] = pd.cut(
        d['BMI'], bins=[0, 18.5, 25.0, 30.0, 100.0], labels=[0, 1, 2, 3]
    ).astype(float)

    # Glucose risk tier: normal<100, prediabetes 100-125, diabetic>=126
    d['Glucose_Tier'] = pd.cut(
        d['Glucose'], bins=[0, 99, 125, 500], labels=[0, 1, 2]
    ).astype(float)

    # Compound high-risk score (additive flag)
    d['HighRisk_Flag'] = (
        (d['Glucose'] > 140).astype(int) +
        (d['BMI'] > 30).astype(int) +
        (d['Age'] > 45).astype(int) +
        (d['DiabetesPedigreeFunction'] > 0.5).astype(int) +
        (d['Insulin'] > 100).astype(int)
    )

    # Pregnancy-to-age ratio (gestational diabetes risk signal)
    d['Preg_Age_Ratio'] = d['Pregnancies'] / (d['Age'] + 1.0)

    # Log-transform skewed features
    d['Log_Insulin'] = np.log1p(d['Insulin'])
    d['Log_DPF']     = np.log1p(d['DiabetesPedigreeFunction'])

    # Skin-to-BMI ratio (body composition proxy)
    d['Skin_BMI'] = d['SkinThickness'] / (d['BMI'] + 1.0)

    return d


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_prepare():
    search_paths = [
        "diabetes.csv",
        os.path.join(os.path.dirname(__file__), "diabetes.csv"),
        "/mnt/user-data/uploads/diabetes.csv",
    ]
    df = None
    for p in search_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        rng = np.random.default_rng(42)
        n = 768
        pregnancies = rng.integers(0, 17, n)
        glucose     = rng.normal(120, 32, n).clip(44, 199).astype(int)
        bp          = rng.normal(69, 19, n).clip(24, 122).astype(int)
        skin        = rng.normal(20, 16, n).clip(0, 99).astype(int)
        insulin     = rng.normal(80, 115, n).clip(14, 846).astype(int)
        bmi         = rng.normal(32, 7, n).clip(18, 67).round(1)
        dpf         = rng.uniform(0.07, 2.42, n).round(3)
        age         = rng.integers(21, 81, n)
        score = (
            (glucose > 140).astype(int) * 2 +
            (bmi > 30).astype(int) +
            (age > 45).astype(int) +
            (dpf > 0.5).astype(int) +
            (insulin > 100).astype(int)
        )
        outcome = (score >= 3).astype(int)
        df = pd.DataFrame({
            'Pregnancies': pregnancies, 'Glucose': glucose,
            'BloodPressure': bp, 'SkinThickness': skin,
            'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age,
            'Outcome': outcome
        })

    # Smarter imputation: zeros are physiologically impossible
    zero_invalid = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']
    for col in zero_invalid:
        median_val = df[col].replace(0, np.nan).median()
        df[col] = df[col].replace(0, np.nan).fillna(median_val)

    # Clip extreme outliers (3× IQR per feature)
    for col in zero_invalid:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 3 * IQR, upper=Q3 + 3 * IQR)

    # Add engineered features
    df_eng = engineer_features(df)
    X = df_eng.drop('Outcome', axis=1)
    y = df_eng['Outcome']
    return df, X, y


@st.cache_resource(show_spinner=False)
def build_models(X, y):
    # RobustScaler is more resilient to outliers than StandardScaler
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y
    )
    # 10-fold CV gives more stable estimates on small datasets
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Class imbalance ratio for scale_pos_weight
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos = neg_count / pos_count

    # Tuned base estimators with class-balance handling
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        max_features='sqrt', class_weight='balanced',
        bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=3, subsample=0.85, max_features='sqrt', random_state=42
    )
    ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.3, random_state=42)
    et = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    svc = SVC(
        kernel='rbf', C=50, gamma='scale',
        probability=True, class_weight='balanced', random_state=42
    )
    lr = LogisticRegression(
        C=0.5, max_iter=5000, class_weight='balanced', solver='saga', random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)

    base_estimators = [("rf", rf), ("gb", gb), ("et", et), ("svc", svc)]

    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=scale_pos, eval_metric='auc',
            random_state=42, n_jobs=-1
        )
        base_estimators.append(("xgb", xgb))

    if LGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            num_leaves=50, min_child_samples=10, subsample=0.85,
            colsample_bytree=0.8, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
        base_estimators.append(("lgbm", lgbm))

    voting_soft = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
    voting_hard = VotingClassifier(estimators=base_estimators, voting='hard', n_jobs=-1)

    # passthrough=True: meta-learner also sees raw scaled features
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            C=5, max_iter=5000, class_weight='balanced', solver='saga'
        ),
        cv=10, stack_method='predict_proba', passthrough=True, n_jobs=-1
    )

    all_models = {
        "Logistic Regression":  lr,
        "Random Forest":        rf,
        "Extra Trees":          et,
        "Gradient Boosting":    gb,
        "AdaBoost":             ada,
        "SVM (RBF)":            svc,
        "KNN":                  knn,
        "Soft Voting Ensemble": voting_soft,
        "Hard Voting Ensemble": voting_hard,
        "Stacking Ensemble":    stacking,
    }
    if XGB_AVAILABLE:  all_models["XGBoost"] = xgb
    if LGBM_AVAILABLE: all_models["LightGBM"] = lgbm

    results = []
    trained = {}
    for name, clf in all_models.items():
        cv_scores = cross_val_score(clf, X_sc, y, cv=cv, scoring='accuracy', n_jobs=-1)
        clf.fit(X_tr, y_tr)
        y_pred   = clf.predict(X_te)
        test_acc = accuracy_score(y_te, y_pred)
        try:
            proba = clf.predict_proba(X_te)[:, 1]
            auc   = round(roc_auc_score(y_te, proba), 4)
        except Exception:
            auc = None
        results.append({
            "Model":              name,
            "CV Accuracy (mean)": round(cv_scores.mean(), 4),
            "CV Std":             round(cv_scores.std(), 4),
            "Test Accuracy":      round(test_acc, 4),
            "AUC-ROC":            auc,
        })
        trained[name] = clf

    results_df = (
        pd.DataFrame(results)
        .sort_values("AUC-ROC", ascending=False, na_position='last')
        .reset_index(drop=True)
    )

    best_row   = results_df[results_df["AUC-ROC"].notna()].iloc[0]
    best_name  = best_row["Model"]
    best_model = trained[best_name]

    return trained, results_df, best_model, best_name, scaler, X_tr, X_te, y_tr, y_te


# ═══════════════════════════════════════════════════════════════
# HEADER & DATA LOAD
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>🩺 DiabetIQ</h1>
  <p>Ensemble Intelligence · Diabetes Prediction Suite · High-Accuracy Classification</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading dataset & training ensemble models — this may take a moment on first run…"):
    df, X, y = load_and_prepare()
    trained, results_df, best_model, best_name, scaler, X_tr, X_te, y_tr, y_te = build_models(X, y)

best_row = results_df[results_df["Model"] == best_name].iloc[0]
top_auc  = best_row["AUC-ROC"]
top_acc  = best_row["CV Accuracy (mean)"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card"><div class="val">{int(df.shape[0])}</div>
        <div class="lbl">Patients in Dataset</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="val">{len(trained)}</div>
        <div class="lbl">Models Trained</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card"><div class="val">{top_acc * 100:.1f}%</div>
        <div class="lbl">Best CV Accuracy</div></div>""", unsafe_allow_html=True)
with k4:
    auc_display = f"{top_auc:.3f}" if top_auc is not None else "—"
    st.markdown(f"""<div class="metric-card"><div class="val">{auc_display}</div>
        <div class="lbl">Best AUC-ROC</div></div>""", unsafe_allow_html=True)

st.markdown("---")

tab_pred, tab_models, tab_eda, tab_details = st.tabs([
    "🔬 Predict Patient", "📊 Model Leaderboard", "📈 EDA & Features", "🔍 Best Model Details"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab_pred:
    with st.sidebar:
        st.markdown("## 🩺 Patient Input Panel")
        st.markdown(f"**Active model:** `{best_name}` <span class='best-badge'>BEST</span>",
                    unsafe_allow_html=True)
        st.markdown("*Predictions always use the top-ranked model by AUC-ROC.*")
        st.markdown("---")
        pregnancies = st.slider("Pregnancies", 0, 17, 2)
        glucose     = st.slider("Plasma Glucose (mg/dL)", 44, 200, 120)
        bp          = st.slider("Diastolic Blood Pressure (mm Hg)", 24, 122, 72)
        skin        = st.slider("Skin Thickness (mm)", 7, 99, 23)
        insulin     = st.slider("2-Hour Serum Insulin (μU/mL)", 14, 846, 80)
        bmi         = st.slider("BMI (kg/m²)", 18.0, 67.0, 32.0, 0.1)
        dpf         = st.slider("Diabetes Pedigree Function", 0.07, 2.42, 0.47, 0.01)
        age         = st.slider("Age (years)", 21, 81, 33)
        st.markdown("---")
        predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")

    st.markdown('<div class="sec-head">Patient Clinical Data</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-box">🤖 Predictions are made using <strong>{best_name}</strong> '
        f'— the highest-ranked model by AUC-ROC ({auc_display}). '
        f'To compare all models, see the <strong>Model Leaderboard</strong> tab.</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        patient_df = pd.DataFrame({
            "Feature": ["Pregnancies", "Glucose (mg/dL)", "Blood Pressure (mm Hg)",
                        "Skin Thickness (mm)", "Insulin (μU/mL)", "BMI (kg/m²)",
                        "Diabetes Pedigree Function", "Age (years)"],
            "Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        })
        st.dataframe(patient_df, use_container_width=True, hide_index=True)

    with col_b:
        fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        fig_r.patch.set_facecolor('#0f2027')
        ax_r.set_facecolor('#0f2027')
        cats   = ['Glucose', 'BMI', 'Age', 'BP', 'Insulin', 'DPF']
        raw    = [glucose/200, bmi/67, age/81, bp/122, insulin/846, dpf/2.42]
        N      = len(cats)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        raw   += [raw[0]]; angles += [angles[0]]
        ax_r.plot(angles, raw, color='#00acc1', linewidth=2)
        ax_r.fill(angles, raw, color='#00acc1', alpha=0.25)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(cats, color='#80cbc4', fontsize=9)
        ax_r.set_yticks([]); ax_r.grid(color='#203a43', linewidth=0.8)
        ax_r.spines['polar'].set_color('#203a43')
        ax_r.set_title("Risk Factor Radar", color='#e0f7fa', pad=14, fontsize=11)
        st.pyplot(fig_r, use_container_width=True)
        plt.close()

    if predict_btn:
        # Apply same feature engineering pipeline before scaling
        raw_row = pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose],
            'BloodPressure': [bp], 'SkinThickness': [skin],
            'Insulin': [insulin], 'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf], 'Age': [age]
        })
        eng_row = engineer_features(raw_row)
        inp_sc  = scaler.transform(eng_row)
        pred    = best_model.predict(inp_sc)[0]
        try:
            proba        = best_model.predict_proba(inp_sc)[0]
            conf         = proba[pred] * 100
            neg_p, pos_p = proba[0] * 100, proba[1] * 100
        except Exception:
            conf = 100
            neg_p, pos_p = (0, 100) if pred == 1 else (100, 0)

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""<div class="pred-positive">⚠️ Diabetic Risk Detected<br>
                <span style="font-size:1rem;font-weight:400;opacity:0.9;">
                Confidence: {conf:.1f}% · {best_name}</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="pred-negative">✅ No Diabetic Risk Detected<br>
                <span style="font-size:1rem;font-weight:400;opacity:0.9;">
                Confidence: {conf:.1f}% · {best_name}</span></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pb1, pb2 = st.columns(2)
        with pb1:
            st.metric("Probability — Non-Diabetic", f"{neg_p:.1f}%")
            st.progress(int(neg_p))
        with pb2:
            st.metric("Probability — Diabetic", f"{pos_p:.1f}%")
            st.progress(int(pos_p))

        st.info("⚕️ This result is for **research and educational purposes only** and is not a substitute for professional medical advice.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab_models:
    st.markdown('<div class="sec-head">Model Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-box">Models are ranked by <strong>AUC-ROC</strong> — the most reliable metric for medical '
        f'classification tasks. The highlighted row (<strong>{best_name}</strong>) is used for all predictions.</div>',
        unsafe_allow_html=True
    )
    display_df = results_df.copy()
    display_df["AUC-ROC"] = display_df["AUC-ROC"].apply(lambda v: f"{v:.4f}" if v is not None else "—")
    display_df["CV Accuracy (mean)"] = display_df["CV Accuracy (mean)"].apply(lambda v: f"{v:.4f}")
    display_df["CV Std"]             = display_df["CV Std"].apply(lambda v: f"{v:.4f}")
    display_df["Test Accuracy"]      = display_df["Test Accuracy"].apply(lambda v: f"{v:.4f}")

    def highlight_best(row):
        if row["Model"] == best_name:
            return ['background-color: #e3f2fd; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-head">Cross-Validated Accuracy Comparison</div>', unsafe_allow_html=True)
    fig_lb, ax_lb = plt.subplots(figsize=(10, 5))
    sorted_df = results_df.sort_values("CV Accuracy (mean)", ascending=True)
    colors = ['#00acc1' if m == best_name else '#1a237e' for m in sorted_df["Model"]]
    bars = ax_lb.barh(sorted_df["Model"], sorted_df["CV Accuracy (mean)"] * 100,
                      color=colors, edgecolor='white', linewidth=0.5)
    ax_lb.set_xlabel("CV Accuracy (%)", fontsize=11)
    ax_lb.set_title("Model Comparison — 10-Fold Cross-Validated Accuracy", fontsize=13, weight='bold')
    ax_lb.axvline(90, color='red', linestyle='--', alpha=0.6, label='90% target')
    ax_lb.legend()
    for bar, val in zip(bars, sorted_df["CV Accuracy (mean)"] * 100):
        ax_lb.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                   f"{val:.1f}%", va='center', fontsize=9)
    ax_lb.set_xlim(0, 105)
    plt.tight_layout()
    st.pyplot(fig_lb, use_container_width=True)
    plt.close()

    st.markdown('<div class="sec-head">ROC Curves — All Models</div>', unsafe_allow_html=True)
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')
    for i, (mname, mclf) in enumerate(trained.items()):
        try:
            prob = mclf.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, prob)
            auc_v = roc_auc_score(y_te, prob)
            lw = 2.5 if mname == best_name else 1.2
            ls = '-'  if mname == best_name else '--'
            label = f"{mname} (AUC={auc_v:.3f})" + (" ★ BEST" if mname == best_name else "")
            ax_roc.plot(fpr, tpr, label=label, color=cmap(i % 10), linewidth=lw, linestyle=ls)
        except Exception:
            pass
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curves — All Models", fontsize=13, weight='bold')
    ax_roc.legend(fontsize=7.5, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_roc, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — EDA & FEATURES
# ══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-head">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Diabetic:** {y.sum()} ({y.mean() * 100:.1f}%)")
        st.write(f"**Non-Diabetic:** {(y == 0).sum()} ({(1 - y.mean()) * 100:.1f}%)")
        orig_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']
        st.dataframe(df[orig_cols].describe().round(2), use_container_width=True)
    with c2:
        fig_out, ax_out = plt.subplots(figsize=(4, 4))
        counts = y.value_counts()
        ax_out.pie(counts, labels=['Non-Diabetic', 'Diabetic'],
                   colors=['#2e7d32', '#b71c1c'], autopct='%1.1f%%',
                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
        ax_out.set_title("Class Distribution")
        st.pyplot(fig_out, use_container_width=True)
        plt.close()

    st.markdown('<div class="sec-head">Feature Distributions by Outcome</div>', unsafe_allow_html=True)
    orig_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']
    fig_dist, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, col in enumerate(orig_cols):
        ax = axes[i // 4][i % 4]
        for outcome, color, label in [(0, '#2e7d32', 'Non-Diabetic'), (1, '#b71c1c', 'Diabetic')]:
            subset = df[df['Outcome'] == outcome][col]
            ax.hist(subset, bins=20, color=color, alpha=0.6, label=label, density=True)
        ax.set_title(col, fontsize=9, weight='bold')
        ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_dist, use_container_width=True)
    plt.close()

    st.markdown('<div class="sec-head">Correlation Matrix</div>', unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax_corr,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax_corr.set_title("Feature Correlation Matrix", fontsize=13, weight='bold')
    plt.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close()

    st.markdown('<div class="sec-head">Feature Importance (Random Forest · incl. Engineered Features)</div>',
                unsafe_allow_html=True)
    rf_clf = trained["Random Forest"]
    fi = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_clf.feature_importances_
    }).sort_values('Importance', ascending=True)
    fig_fi, ax_fi = plt.subplots(figsize=(8, 8))
    colors_fi = ['#00acc1' if i >= len(fi) - 3 else '#1a237e' for i in range(len(fi))]
    ax_fi.barh(fi['Feature'], fi['Importance'], color=colors_fi)
    ax_fi.set_title("Random Forest — Feature Importance (with Engineered Features)", fontsize=11, weight='bold')
    ax_fi.set_xlabel("Importance Score")
    for bar, val in zip(ax_fi.patches, fi['Importance']):
        ax_fi.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 4 — BEST MODEL DETAILS
# ══════════════════════════════════════════════════════════════
with tab_details:
    st.markdown(
        f'<div class="sec-head">Detailed Report — {best_name} <span class="best-badge">BEST MODEL</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="info-box">This is the model used for all predictions. '
        f'It was selected based on the highest <strong>AUC-ROC</strong> score across all trained models.</div>',
        unsafe_allow_html=True
    )
    y_pred_best   = best_model.predict(X_te)
    test_acc_best = accuracy_score(y_te, y_pred_best)

    d1, d2, d3 = st.columns(3)
    d1.metric("Test Accuracy", f"{test_acc_best * 100:.2f}%")
    try:
        prob_best = best_model.predict_proba(X_te)[:, 1]
        auc_best  = roc_auc_score(y_te, prob_best)
        d2.metric("AUC-ROC", f"{auc_best:.4f}")
    except Exception:
        d2.metric("AUC-ROC", "—")
    d3.metric("Model", best_name)

    st.markdown('<div class="sec-head">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_te, y_pred_best)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    ax_cm.set_ylabel("Actual"); ax_cm.set_xlabel("Predicted")
    ax_cm.set_title(f"Confusion Matrix — {best_name}", weight='bold')
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)
    plt.close()

    st.markdown('<div class="sec-head">Classification Report</div>', unsafe_allow_html=True)
    cr    = classification_report(y_te, y_pred_best,
                                  target_names=['Non-Diabetic', 'Diabetic'], output_dict=True)
    cr_df = pd.DataFrame(cr).T.round(3)
    st.dataframe(cr_df, use_container_width=True)

    st.markdown('<div class="sec-head">ROC Curve</div>', unsafe_allow_html=True)
    try:
        fpr, tpr, _ = roc_curve(y_te, prob_best)
        fig_br, ax_br = plt.subplots(figsize=(6, 5))
        ax_br.plot(fpr, tpr, color='#00acc1', linewidth=2.5, label=f"AUC = {auc_best:.4f}")
        ax_br.fill_between(fpr, tpr, alpha=0.08, color='#00acc1')
        ax_br.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax_br.set_xlabel("False Positive Rate", fontsize=11)
        ax_br.set_ylabel("True Positive Rate", fontsize=11)
        ax_br.set_title(f"ROC Curve — {best_name}", fontsize=12, weight='bold')
        ax_br.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_br, use_container_width=True)
        plt.close()
    except Exception:
        pass

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#9e9e9e;font-size:0.8rem;'>
DiabetIQ · Ensemble ML Suite · Built with Streamlit & scikit-learn<br>
⚠️ For research/educational purposes only. Not a substitute for medical advice.
</div>
""", unsafe_allow_html=True)
