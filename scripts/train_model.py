# ======================================================
# 🩺 Diabetes Risk Prediction & Statistical Analysis Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction - Statistical Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction & Statistical Analysis")
st.markdown("""
This dashboard combines **Machine Learning** and **Statistical Analysis**  
to understand diabetes risk factors using real-world data.
""")

# ------------------------ 1. LOAD DATA ------------------------
data = pd.read_csv("diabetes10k.csv")

st.subheader("📂 Dataset Overview")
st.dataframe(data.head())

# ------------------------ 2. DESCRIPTIVE STATISTICS ------------------------
st.subheader("📈 Descriptive Statistics")
st.dataframe(data.describe().T)

# ------------------------ 3. CORRELATION HEATMAP ------------------------
st.subheader("🔗 Feature Correlation Heatmap")

# Only use numeric columns for correlation
numeric_df = data.select_dtypes(include=[np.number])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ------------------------ 4. TRAIN-TEST SPLIT & SCALING ------------------------
X = data.drop(["Outcome", "Country"], axis=1, errors='ignore')
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ 5. TRAIN MODELS ------------------------
log_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)

# ------------------------ 6. EVALUATION ------------------------
y_pred = log_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.subheader("🎯 Model Performance (Logistic Regression)")
st.metric(label="Accuracy", value=f"{acc*100:.2f}%")

# Cross-validation
scores = cross_val_score(log_model, X, y, cv=5)
mean_acc = np.mean(scores)
conf_interval = 1.96 * np.std(scores) / np.sqrt(len(scores))

st.write(f"**Cross-Validation Accuracy:** {mean_acc:.3f}")
st.write(f"**95% Confidence Interval:** ±{conf_interval:.3f}")

# ------------------------ 7. HYPOTHESIS TESTING ------------------------
st.subheader("🧪 Hypothesis Testing")

# Compare glucose levels between diabetic and non-diabetic patients
group1 = data[data['Outcome'] == 0]['Glucose']
group2 = data[data['Outcome'] == 1]['Glucose']

t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

st.write(f"**T-statistic:** {t_stat:.4f}")
st.write(f"**P-value:** {p_val:.6f}")

if p_val < 0.05:
    st.success("✅ Statistically significant difference in Glucose levels between diabetic and non-diabetic groups.")
else:
    st.warning("❌ No significant difference found.")

# Boxplot Visualization
fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.boxplot(x='Outcome', y='Glucose', data=data, palette='Set2', ax=ax2)
ax2.set_title("Distribution of Glucose Levels by Diabetes Outcome")
st.pyplot(fig2)

# ------------------------ 8. SAVE MODEL ------------------------
joblib.dump(log_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

st.info("💾 Model and scaler saved successfully.")
st.success("✅ Statistical + Predictive Analysis Complete.")
