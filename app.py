# app.py
# Single Streamlit app: retrain models + statistical analysis + prediction UI

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from scipy import stats

# VIF
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATS_MODELS_AVAILABLE = True
except Exception:
    STATS_MODELS_AVAILABLE = False

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Diabetes Risk - Statistical Dashboard", layout="wide")
st.title("🩺 Diabetes Risk Prediction — Statistical Dashboard (Retrain Mode)")

# ----------------------------
# Helper: load dataset
# ----------------------------
def load_data():
    # prefer data/diabetes10k.csv if present
    paths = ["data/diabetes10k.csv", "diabetes10k.csv", "data/diabetes.csv", "diabetes.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("No dataset found. Put 'diabetes10k.csv' in data/ or project root.")
    st.stop()

data = load_data()

# show dataset basic info
with st.expander("📂 Dataset preview and info", expanded=True):
    st.write("Shape:", data.shape)
    st.dataframe(data.head())

# ----------------------------
# Clean / Preprocess
# ----------------------------
def preprocess(df):
    df = df.copy()
    # Drop non-numeric columns that aren't features (Country, ID, etc.)
    # Keep Outcome column if present
    # auto-detect and drop object columns except Outcome if Outcome is object (rare)
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])
    # Ensure Outcome exists
    if "Outcome" not in df.columns:
        st.error("Dataset must contain 'Outcome' column (0/1).")
        st.stop()
    # Convert non-numeric columns (if any) - drop them for modelling (or encode)
    non_numeric = df.select_dtypes(include=['object','category']).columns.tolist()
    non_numeric = [c for c in non_numeric if c != "Outcome"]
    if non_numeric:
        st.warning(f"Dropping non-numeric columns for modelling: {non_numeric}")
        df = df.drop(columns=non_numeric)
    df = df.dropna()  # simple drop rows with NaNs; you can improve later
    return df

df = preprocess(data)

# features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# sidebar controls
st.sidebar.header("Controls & Options")
retrain = st.sidebar.button("🔁 Retrain models now")
use_cached = st.sidebar.checkbox("Use cached models if available", value=True)
cv_folds = st.sidebar.number_input("Cross-val folds", min_value=3, max_value=10, value=5, step=1)
test_size = st.sidebar.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# ----------------------------
# Preprocess: scaling
# ----------------------------
def get_train_test_scaled(X, y, test_size=test_size, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_train_test_scaled(X, y, test_size=test_size)

# ----------------------------
# Train models
# ----------------------------
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def train_models(X_train_scaled, y_train):
    # Logistic
    log = LogisticRegression(max_iter=1000, random_state=42)
    log.fit(X_train_scaled, y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # XGBoost (if available)
    xgb = None
    if XGB_AVAILABLE:
        xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        xgb.fit(X_train_scaled, y_train)

    return log, rf, xgb

# caching: if models exist and use_cached, load them unless retrain requested
def load_cached_models():
    log_path = os.path.join(models_dir,"logistic.pkl")
    rf_path = os.path.join(models_dir,"rf.pkl")
    xgb_path = os.path.join(models_dir,"xgb.pkl")
    scaler_path = os.path.join(models_dir,"scaler.pkl")
    if os.path.exists(log_path) and os.path.exists(rf_path) and os.path.exists(scaler_path):
        log = joblib.load(log_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        xgb = None
        if XGB_AVAILABLE and os.path.exists(xgb_path):
            xgb = joblib.load(xgb_path)
        return log, rf, xgb, scaler
    return None

cached = None
if use_cached and not retrain:
    cached = load_cached_models()

if cached is not None and not retrain:
    log_model, rf_model, xgb_model, loaded_scaler = cached
    st.sidebar.success("Loaded cached models.")
    # ensure scaler matches current split feature order; we saved original scaler, so we will overwrite scaler var to loaded_scaler
    scaler = loaded_scaler
else:
    with st.spinner("Training models... (this may take a moment)"):
        log_model, rf_model, xgb_model = train_models(X_train_scaled, y_train)
        # save artifacts
        joblib.dump(log_model, os.path.join(models_dir,"logistic.pkl"))
        joblib.dump(rf_model, os.path.join(models_dir,"rf.pkl"))
        if xgb_model is not None:
            joblib.dump(xgb_model, os.path.join(models_dir,"xgb.pkl"))
        joblib.dump(scaler, os.path.join(models_dir,"scaler.pkl"))
    st.sidebar.success("Models trained and saved to /models/")

# ----------------------------
# Evaluate & Cross-val + CI
# ----------------------------
st.header("Model evaluation & statistical reliability")

def crossval_and_ci(model, X_full, y_full, cv=cv_folds):
    scores = cross_val_score(model, X_full, y_full, cv=cv)
    mean_acc = np.mean(scores)
    ci = 1.96 * np.std(scores) / np.sqrt(len(scores))
    return mean_acc, ci, scores

# Because cross_val_score expects raw X (not scaled) we'll scale inside cross_val using a pipeline, but for simplicity we reuse scaled arrays:
# We'll compute CV on full X after scaling with StandardScaler fitted on full X to avoid data leakage in this demonstration:
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)

cv_results = {}
for name, model in [("Logistic", log_model), ("RandomForest", rf_model), ("XGBoost", xgb_model)]:
    if model is None:
        continue
    mean_acc, ci, scores = crossval_and_ci(model, X_full_scaled, y, cv=cv_folds)
    cv_results[name] = {"mean_acc": mean_acc, "ci": ci, "scores": scores}
    st.write(f"**{name}** — CV mean accuracy: {mean_acc:.3f}  |  95% CI ±{ci:.3f}")

# Show simple comparison bar chart
comp_df = pd.DataFrame({
    name: [cv_results[name]["mean_acc"]] for name in cv_results
}).T
comp_df.columns = ["mean_acc"]
st.bar_chart(comp_df)

# ----------------------------
# Confusion matrix and ROC for logistic on test set
# ----------------------------
st.subheader("Logistic Regression — test performance & ROC curve")
y_pred = log_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy (Logistic): {test_acc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
ax.set_title("Confusion Matrix (Logistic)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ROC curves for available models
st.subheader("ROC Curves (available models)")
fig2, ax2 = plt.subplots(figsize=(6,4))
for name, model in [("Logistic", log_model), ("RandomForest", rf_model), ("XGBoost", xgb_model)]:
    if model is None:
        continue
    # need probabilities or decision_function
    try:
        probs = model.predict_proba(X_test_scaled)[:,1]
    except Exception:
        # fallback to decision_function
        try:
            probs = model.decision_function(X_test_scaled)
            probs = (probs - probs.min())/(probs.max()-probs.min())
        except Exception:
            continue
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
ax2.plot([0,1],[0,1], 'k--', linewidth=0.7)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves")
ax2.legend()
st.pyplot(fig2)

# ----------------------------
# Statistical analysis: descriptive stats, correlation, t-test
# ----------------------------
st.header("Statistical analysis & interpretation")

# Descriptive stats
st.subheader("Descriptive statistics")
st.dataframe(df.describe().T)

# Correlation heatmap (numeric only)
st.subheader("Correlation heatmap (numeric features only)")
numeric_df = df.select_dtypes(include=[np.number])
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Hypothesis testing: t-test for Glucose if present
if "Glucose" in df.columns:
    st.subheader("Hypothesis test example: Glucose levels by Outcome")
    g0 = df[df["Outcome"]==0]["Glucose"]
    g1 = df[df["Outcome"]==1]["Glucose"]
    t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
    st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.6f}")
    if p_val < 0.05:
        st.success("Significant difference in Glucose between groups (p < 0.05).")
    else:
        st.info("No significant difference found.")
    # Boxplot
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.boxplot(x="Outcome", y="Glucose", data=df, ax=ax4)
    ax4.set_title("Glucose by Outcome")
    st.pyplot(fig4)
else:
    st.info("No 'Glucose' column found; skipping glucose t-test.")

# ----------------------------
# Logistic regression coefficients & odds ratios
# ----------------------------
st.subheader("Logistic Regression coefficients & Odds Ratios (interpretation)")

# Fit logistic on full training scaled features for coefficient interpretation
log_for_interpret = LogisticRegression(max_iter=1000, random_state=42)
log_for_interpret.fit(scaler_full.fit_transform(X), y)  # fit on full dataset scaled

coeffs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": log_for_interpret.coef_[0],
    "odds_ratio": np.exp(log_for_interpret.coef_[0])
}).sort_values(by="odds_ratio", ascending=False)

st.dataframe(coeffs.style.format({"coefficient":"{:.4f}", "odds_ratio":"{:.3f}"}))
st.write("Interpretation example: 'Odds Ratio = 1.50' means a 1-unit increase in the feature multiplies odds by 1.50x.")

# ----------------------------
# VIF (multicollinearity)
# ----------------------------
if STATS_MODELS_AVAILABLE:
    st.subheader("Variance Inflation Factor (VIF)")
    X_const = sm.add_constant(X)
    vif_data = []
    for i, col in enumerate(X.columns):
        vif_val = variance_inflation_factor(X_const.values, i+1)
        vif_data.append((col, vif_val))
    vif_df = pd.DataFrame(vif_data, columns=["feature","VIF"]).sort_values(by="VIF", ascending=False)
    st.dataframe(vif_df)
    st.write("VIF > 5 (or >10) indicates potential multicollinearity problems.")
else:
    st.warning("statsmodels not installed — VIF calculation skipped. Install statsmodels to enable VIF.")

# ----------------------------
# Interactive prediction panel
# ----------------------------
st.header("Interactive prediction (use trained models)")

with st.form("predict_form"):
    st.write("Enter feature values (leave blank to use dataset mean):")
    inputs = {}
    for col in X.columns:
        col_min = float(X[col].min()); col_max = float(X[col].max()); col_mean = float(X[col].mean())
        val = st.number_input(label=f"{col}", value=col_mean, format="%.4f")
        inputs[col] = val
    model_choice = st.selectbox("Choose model for prediction", options=[m for m in ["Logistic","RandomForest","XGBoost"] if not (m=="XGBoost" and not XGB_AVAILABLE)])
    submit = st.form_submit_button("Predict")

if submit:
    sample = pd.DataFrame([inputs])
    sample_scaled = scaler.transform(sample)
    chosen_model = None
    if model_choice == "Logistic":
        chosen_model = log_model
    elif model_choice == "RandomForest":
        chosen_model = rf_model
    elif model_choice == "XGBoost" and XGB_AVAILABLE:
        chosen_model = xgb_model
    if chosen_model is None:
        st.error("Chosen model not available.")
    else:
        try:
            prob = chosen_model.predict_proba(sample_scaled)[0,1]
        except Exception:
            # fallback
            pred = chosen_model.predict(sample_scaled)[0]
            prob = float(pred)
        st.write(f"Predicted probability of diabetes (model={model_choice}): **{prob:.3f}**")
        label = 1 if prob >= 0.5 else 0
        st.write(f"Predicted label (threshold 0.5): **{label}**")

# ----------------------------
# Save final artifacts info
# ----------------------------
st.info(f"Trained models are saved to the `{models_dir}/` directory (logistic.pkl, rf.pkl, xgb.pkl if xgboost available, scaler.pkl).")

st.markdown("---")
st.caption("This dashboard retrains models and provides statistical explanations for a statistics project. For submission, include the generated tables (coefficients, VIF, CV scores) and screenshots of the dashboard.")
