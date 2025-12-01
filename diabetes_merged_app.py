# Unified Streamlit app: diabetes_merged_app.py
# Contains: data loading, statistical analysis, model training/comparison,
# prediction UI, explainability (SHAP if available), and advanced statistical tests

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import resample
from xgboost import XGBClassifier

# Optional: SHAP - for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ----------------------------- CONFIG ---------------------------------
APP_TITLE = "Diabetes Risk Prediction — Advanced"
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSVS = [
    Path("data/diabetes10k.csv"),
    Path("diabetes10k.csv"),
    Path("data/diabetes.csv"),
    Path("diabetes.csv"),
]

DEFAULT_MODEL = MODEL_DIR / "diabetes_model.pkl"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"

# ----------------------------- UTILITIES ---------------------------------
@st.cache_data
def load_csv_from_candidates(candidates=DEFAULT_CSVS):
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data
def read_data(path: Path):
    return pd.read_csv(path)

def safe_save_joblib(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def safe_load_joblib(path: Path):
    if path.exists():
        return joblib.load(path)
    return None

# ----------------------------- PREPROCESS ---------------------------------
@st.cache_data
def basic_preprocess(df: pd.DataFrame, target_col: str = "Outcome"):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Coerce numeric-like strings
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c].str.replace(',', ''), errors='coerce')
            except Exception:
                pass

    # Numeric imputation
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Drop remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    return X, y

# ----------------------------- STATISTICAL HELPERS -------------------------------
def outlier_counts(df: pd.DataFrame):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    return outliers

def skew_kurtosis(df: pd.DataFrame):
    return pd.DataFrame({"skew": df.skew(), "kurtosis": df.kurt()})

def chi2_categorical_assoc(df: pd.DataFrame, cat_cols, target_col='Outcome'):
    results = []
    for c in cat_cols:
        tbl = pd.crosstab(df[c].fillna('NA'), df[target_col])
        try:
            chi2, p, dof, ex = stats.chi2_contingency(tbl)
            results.append({"feature": c, "chi2": float(chi2), "p_value": float(p)})
        except Exception:
            results.append({"feature": c, "chi2": None, "p_value": None})
    return pd.DataFrame(results)

def anova_ttest_numeric(df: pd.DataFrame, numeric_cols, target_col='Outcome'):
    rows = []
    for c in numeric_cols:
        g0 = df[df[target_col]==0][c].dropna()
        g1 = df[df[target_col]==1][c].dropna()
        try:
            tstat, p = stats.ttest_ind(g0, g1, equal_var=False)
            rows.append({"feature": c, "t_stat": float(tstat), "p_value": float(p)})
        except Exception:
            rows.append({"feature": c, "t_stat": None, "p_value": None})
    return pd.DataFrame(rows)

def compute_vif(X: pd.DataFrame):
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df

# ----------------------------- MODEL COMPARISON ---------------------------------
@st.cache_data
def compare_models_cv(X, y, cv=5, random_state=42):
    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))]),
        "RandomForest": Pipeline([("model", RandomForestClassifier(n_estimators=200, random_state=random_state))]),
        "XGBoost": Pipeline([("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))]),
        "SVC": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=random_state))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    }
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X, y, scoring=scoring, cv=skf, n_jobs=-1)
        row = {"model": name}
        for s in scoring:
            row[f"{s}_mean"] = float(np.mean(res[f"test_{s}"]))
            row[f"{s}_std"] = float(np.std(res[f"test_{s}"]))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(by="roc_auc_mean", ascending=False).reset_index(drop=True)
    try:
        df.to_csv(COMPARISON_CSV, index=False)
    except Exception:
        pass
    return df

# ----------------------------- TRAIN & SAVE -------------------------------
def train_and_save_best(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    candidates = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))]),
        "RandomForest": Pipeline([("model", RandomForestClassifier(n_estimators=200, random_state=random_state))]),
        "XGBoost": Pipeline([("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))]),
    }
    best_model = None
    best_auc = -1
    stats_rows = []
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:,1]
        else:
            try:
                probs = pipe.decision_function(X_test)
            except Exception:
                probs = np.zeros_like(y_test)
        preds = pipe.predict(X_test)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.0
        acc = accuracy_score(y_test, preds)
        stats_rows.append({"model": name, "accuracy": float(acc), "auc": float(auc)})
        if auc > best_auc:
            best_auc = auc
            best_model = pipe
    if best_model is not None:
        safe_save_joblib(best_model, DEFAULT_MODEL)
    return best_model, pd.DataFrame(stats_rows).sort_values("auc", ascending=False)

# ----------------------------- EXPLAINABILITY (SHAP) -------------------------------
@st.cache_data
def compute_shap_explain(model, X_sample):
    if not SHAP_AVAILABLE:
        return None
    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            X_trans = model.named_steps.get("scaler", None)
            if X_trans:
                X_trans = model.named_steps["scaler"].transform(X_sample)
            else:
                X_trans = X_sample.values
        else:
            estimator = model
            X_trans = X_sample.values
        explainer = shap.Explainer(estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict, X_trans)
        shap_values = explainer(X_trans)
        return shap_values
    except Exception:
        return None

# ----------------------------- ADVANCED STATISTICAL UI -------------------------------
def show_header():
    st.markdown(f"# {APP_TITLE}")
    st.markdown("---")

def sidebar_inputs():
    st.sidebar.header("Data & Model Options")
    uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
    use_default = st.sidebar.checkbox("Use default dataset (if present in /data)", value=True)
    st.sidebar.markdown("---")
    st.sidebar.header("Model Files (optional)")
    st.sidebar.text_input("Model path", str(DEFAULT_MODEL))
    return uploaded, use_default

def show_data_preview(df):
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    with st.expander("Show full dataframe info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write(df.describe())

def show_statistical_analysis(df):
    st.subheader("Statistical Analysis & Tests")

    numeric_df = df.select_dtypes(include=["number"])
    cat_df = df.select_dtypes(include=['object', 'category'])

    # Basic stats
    if st.checkbox("Show summary statistics (describe)", value=True):
        st.write(numeric_df.describe().T)

    # Skewness & Kurtosis
    if st.checkbox("Show skewness & kurtosis"):
        sk = skew_kurtosis(numeric_df)
        st.dataframe(sk.sort_values('skew', key=lambda s: s.abs(), ascending=False))

    # Outliers count
    if st.checkbox("Show outlier counts (IQR method)"):
        if numeric_df.shape[1] >= 1:
            oc = outlier_counts(numeric_df)
            st.write(oc.sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=numeric_df[numeric_df.columns[:6]])
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for outlier analysis.")

    # Correlation heatmaps: Pearson + Spearman
    if st.checkbox("Show correlation heatmaps"):
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(numeric_df.corr(method='pearson'), annot=True, fmt='.2f', cmap='coolwarm', ax=ax[0])
            ax[0].set_title('Pearson correlation')
            sns.heatmap(numeric_df.corr(method='spearman'), annot=True, fmt='.2f', cmap='vlag', ax=ax[1])
            ax[1].set_title('Spearman correlation')
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns to compute correlations.")

    # Chi-square for categorical features
    if cat_df.shape[1] > 0 and st.checkbox("Run chi-square tests for categorical features"):
        with st.spinner("Running chi-square tests..."):
            chi_df = chi2_categorical_assoc(df, cat_df.columns.tolist(), target_col='Outcome' if 'Outcome' in df.columns else df.columns[-1])
            st.dataframe(chi_df)

    # t-test / ANOVA for numeric features by Outcome
    if 'Outcome' in df.columns and st.checkbox("Run t-tests (numeric features vs Outcome)"):
        numeric_cols = numeric_df.columns.tolist()
        tdf = anova_ttest_numeric(df, numeric_cols, target_col='Outcome')
        st.dataframe(tdf.sort_values('p_value'))

    # VIF
    if st.checkbox("Compute VIF (multicollinearity)"):
        if numeric_df.shape[1] >= 2:
            try:
                vif = compute_vif(numeric_df)
                st.dataframe(vif.sort_values('VIF', ascending=False))
            except Exception as e:
                st.error(f"VIF computation failed: {e}")
        else:
            st.info("Not enough numeric variables for VIF.")

    st.markdown("---")
    st.write("**Optional advanced visual tools (PDP, PCA, clustering) — use only if dataset size is moderate.**")

    # PCA visualization
    if st.checkbox("Run PCA and show 2D projection"):
        if numeric_df.shape[1] >= 2:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(numeric_df.fillna(0))
            pca_df = pd.DataFrame(proj, columns=['PC1', 'PC2'])
            if 'Outcome' in df.columns:
                pca_df['Outcome'] = df['Outcome'].values
            fig, ax = plt.subplots()
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Outcome' if 'Outcome' in pca_df.columns else None, alpha=0.7)
            st.pyplot(fig)
        else:
            st.info("PCA requires at least 2 numeric features.")

    # Clustering
    if st.checkbox("Run kmeans clustering (k=2..6)"):
        if numeric_df.shape[1] >= 2:
            k = st.slider("Choose k (clusters)", 2, 6, 3)
            km = KMeans(n_clusters=k, random_state=42)
            try:
                labels = km.fit_predict(numeric_df.fillna(0))
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=labels, palette='tab10')
                ax.set_xlabel(numeric_df.columns[0]); ax.set_ylabel(numeric_df.columns[1])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"KMeans failed: {e}")
        else:
            st.info("KMeans needs at least 2 numeric features.")

# ----------------------------- MODEL COMPARISON UI -------------------------------

def show_model_comparison_ui(X, y):
    st.subheader("Model Comparison (Cross-Validation)")
    if COMPARISON_CSV.exists():
        st.info(f"Found cached comparison at {COMPARISON_CSV}; load cached results or re-run")
        if st.button("Load cached comparison"):
            dfc = pd.read_csv(COMPARISON_CSV)
            st.dataframe(dfc)
    if st.button("Run CV comparison now (may take a few minutes)"):
        with st.spinner("Running CV across candidates..."):
            dfc = compare_models_cv(X, y, cv=5)
        st.success("Comparison finished")
        st.dataframe(dfc)
        st.download_button("Download comparison CSV", dfc.to_csv(index=False).encode('utf-8'), file_name="model_comparison.csv")

# ----------------------------- MODEL TRAINING UI -------------------------------

def show_training_ui(X, y):
    st.subheader("Train & Save Best Model")
    st.markdown("This trains a few candidate models and saves the best one by ROC AUC to `models/diabetes_model.pkl`.")
    if st.button("Train & Save best model"):
        with st.spinner("Training models..."):
            best, stats = train_and_save_best(X, y)
        if best is not None:
            st.success("Trained and saved best model to models/diabetes_model.pkl")
            st.dataframe(stats)
        else:
            st.error("Training failed; check logs")

# ----------------------------- FEATURE IMPORTANCE & PDP -------------------------------

def show_feature_importance_and_pdp(model, X, y):
    st.subheader("Feature Importance & Partial Dependence")
    try:
        # Permutation importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values('importance', ascending=False)
        st.write("### Permutation Importance")
        st.dataframe(imp_df)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=imp_df.head(10), x='importance', y='feature', ax=ax)
        st.pyplot(fig)

        # Partial Dependence for top 1-2 features
        top_feats = imp_df['feature'].head(2).tolist()
        if top_feats:
            try:
                fig = plt.figure(figsize=(8, 4))
                PartialDependenceDisplay.from_estimator(model, X, top_feats)
                st.pyplot(fig)
            except Exception as e:
                st.write("Partial dependence failed:", e)

    except Exception as e:
        st.write("Feature importance computation failed:", e)

# ----------------------------- MODEL EVALUATION & CALIBRATION -------------------------------

def evaluate_model_on_holdout(model, X_test, y_test):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = np.zeros(len(y_test))
    preds = (probs >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs) if len(np.unique(y_test))>1 else 0.0
    }
    return metrics, preds, probs


def show_evaluation_and_calibration(model, X_test, y_test):
    st.subheader("Model Evaluation & Calibration")
    metrics, preds, probs = evaluate_model_on_holdout(model, X_test, y_test)
    st.write(metrics)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        st.pyplot(fig)
    except Exception as e:
        st.write('Calibration failed:', e)

# ----------------------------- STATISTICAL MODEL COMPARISONS -------------------------------

def paired_model_ttest(model1, model2, X, y, cv=5):
    s1 = cross_val_score(model1, X, y, cv=cv, scoring='roc_auc')
    s2 = cross_val_score(model2, X, y, cv=cv, scoring='roc_auc')
    t, p = stats.ttest_rel(s1, s2)
    return t, p, s1.mean(), s2.mean()

# Bootstrap CI for accuracy
def bootstrap_ci_accuracy(model, X_test, y_test, n_boot=200):
    accs = []
    for i in range(n_boot):
        Xb, yb = resample(X_test, y_test, replace=True)
        if hasattr(model, 'predict'):
            accs.append(accuracy_score(yb, model.predict(Xb)))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

# ----------------------------- PREDICTION UI -------------------------------

def show_prediction_ui(df, feature_order=None):
    st.subheader("Diabetes Risk Prediction — Live")
    model = safe_load_joblib(DEFAULT_MODEL)
    if model is None:
        st.warning("No saved model found at models/diabetes_model.pkl. Use the Train section to create one.")
        return

    st.markdown("Enter patient features below (use same columns as the training data).")
    X_sample = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != 'outcome']

    if feature_order is None:
        feature_order = numeric_cols

    cols = st.columns(3)
    for i, c in enumerate(feature_order):
        with cols[i % 3]:
            minv = float(df[c].min()) if c in df.columns else 0.0
            maxv = float(df[c].max()) if c in df.columns else 100.0
            meanv = float(df[c].median()) if c in df.columns else 0.0
            X_sample[c] = st.number_input(c, min_value=minv, max_value=maxv, value=meanv)

    if st.button("Predict risk"):
        X_df = pd.DataFrame([X_sample])[feature_order]
        try:
            if hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba(X_df)[:,1][0]) * 100
            else:
                prob = float(model.decision_function(X_df)[0])
            st.metric("Diabetes probability (%)", f"{prob:.2f}%")

            thresh = st.slider("Decision threshold", 0.0, 1.0, 0.5)
            label = (prob/100.0 >= thresh)
            st.write("Prediction:", "Diabetic" if label else "Not diabetic")

            if SHAP_AVAILABLE and st.checkbox("Show SHAP explanation for this input"):
                shap_vals = compute_shap_explain(model, X_df)
                if shap_vals is not None:
                    try:
                        st.pyplot(shap.plots.bar(shap_vals, show=False))
                    except Exception:
                        st.write(shap_vals)
                else:
                    st.info("SHAP explanation unavailable for this model.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------- MAIN -----------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    show_header()
    uploaded, use_default = sidebar_inputs()

    # Load dataset
    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("Loaded uploaded dataset")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    else:
        csv_path = load_csv_from_candidates()
        if csv_path and use_default:
            try:
                df = read_data(csv_path)
                st.info(f"Loaded default dataset from {csv_path}")
            except Exception as e:
                st.error(f"Failed to read default CSV: {e}")

    if df is None:
        st.warning("No dataset loaded.")
        return

    # Preview & statistics
    show_data_preview(df)
    show_statistical_analysis(df)

    # Preprocess: numeric features only
    try:
        X, y = basic_preprocess(df, target_col='Outcome')
        feature_order = X.columns.tolist()  # Only numeric features used for prediction
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    # Model comparison and training
    show_model_comparison_ui(X, y)
    show_training_ui(X, y)

    # Evaluate model on holdout (numeric features only)
    model = safe_load_joblib(DEFAULT_MODEL)
    if model:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            show_evaluation_and_calibration(model, X_test, y_test)
            show_feature_importance_and_pdp(model, X_test, y_test)

            if st.checkbox("Run bootstrap CI for current saved model (accuracy)"):
                ci_low, ci_high = bootstrap_ci_accuracy(model, X_test, y_test, n_boot=200)
                st.write(f"95% CI for accuracy: [{ci_low:.3f}, {ci_high:.3f}]")

        except Exception as e:
            st.write("Model evaluation failed:", e)

    # Prediction UI — numeric columns only
    show_prediction_ui(df[X.columns], feature_order=feature_order)
