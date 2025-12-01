import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    roc_auc_score,
)

# ------------- SETTINGS -------------
CSV_PATH = "diabetes10k.csv"
OUT_DIR = "graphs_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- LOAD DATA -------------
df = pd.read_csv(CSV_PATH)

# Assume binary target column is 'Outcome'
y = df["Outcome"].astype(int)
X_raw = df.drop(columns=["Outcome"])

# One-hot encode categorical features (Country etc.)
X = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}

# ------------- TRAIN + EVAL -------------
fpr_dict = {}
tpr_dict = {}
auc_dict = {}
acc_dict = {}
cms = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_dict[name] = fpr
    tpr_dict[name] = tpr
    auc_dict[name] = auc(fpr, tpr)
    acc_dict[name] = (y_pred == y_test).mean()

    cms[name] = confusion_matrix(y_test, y_pred)

# ------------- GRAPH 1: CHRONOLOGICAL EVOLUTION -------------
years = [1988, 2017, 2019, 2020, 2021, 2022, 2023, 2025]
best_acc = [76, 88.7, 90, 89, 88, 90, 90, 89]

plt.figure()
plt.plot(years, best_acc, marker="o")
plt.xlabel("Year")
plt.ylabel("Best Accuracy on PIDD (%)")
plt.title("Chronological Evolution of PIDD Model Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_chronological_evolution.png"))
plt.close()

# ------------- GRAPH 2: LITERATURE ACCURACY BAR CHART -------------
studies = [
    "Smith 1988\nADAP NN",
    "Jahangir 2017\nECO-AMLP",
    "Naz & Ahuja 2020\nDNN",
    "Aslan 2023\nCNN",
    "Stacking 2022",
    "Hybrid GA-MLP 2023",
    "Reza 2023\nSVM",
    "Metaheuristic FS 2019",
]
accs_lit = [76, 88.7, 89, 90, 90, 88, 87, 88]

plt.figure()
plt.barh(studies, accs_lit)
plt.xlabel("Accuracy (%)")
plt.title("Landmark PIDD Model Accuracies")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_literature_accuracy.png"))
plt.close()

# ------------- GRAPH 3: YOUR PROJECT ACCURACY -------------
plt.figure()
names = list(acc_dict.keys())
vals = [acc_dict[n] * 100 for n in names]
plt.bar(names, vals)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy on diabetes10k")
for i, v in enumerate(vals):
    plt.text(i, v + 0.5, f"{v:.1f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_project_accuracy.png"))
plt.close()

# ------------- GRAPH 4: PREPROCESSING IMPACT (LR) -------------
# numeric-only subset
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Outcome" in num_cols:
    num_cols.remove("Outcome")

X_num = df[num_cols]
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
    X_num, y, test_size=0.2, random_state=42, stratify=y
)

lr_raw = LogisticRegression(max_iter=1000)
lr_raw.fit(Xn_tr, yn_tr)
yprob_raw = lr_raw.predict_proba(Xn_te)[:, 1]
auc_raw = roc_auc_score(yn_te, yprob_raw)

sc2 = StandardScaler()
Xn_tr_s = sc2.fit_transform(Xn_tr)
Xn_te_s = sc2.transform(Xn_te)
lr_scaled = LogisticRegression(max_iter=1000)
lr_scaled.fit(Xn_tr_s, yn_tr)
yprob_scaled = lr_scaled.predict_proba(Xn_te_s)[:, 1]
auc_scaled = roc_auc_score(yn_te, yprob_scaled)

plt.figure()
plt.bar(["LR (No Scaling)", "LR (Scaled)"], [auc_raw, auc_scaled])
plt.ylabel("ROC-AUC")
plt.title("Impact of Feature Scaling on Logistic Regression (diabetes10k)")
for i, v in enumerate([auc_raw, auc_scaled]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_preprocessing_impact.png"))
plt.close()

# ------------- GRAPH 5: CORRELATION HEATMAP -------------
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr.values)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap - Numeric Features (diabetes10k)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_correlation_heatmap.png"))
plt.close()

# ------------- GRAPH 6: ROC CURVES -------------
plt.figure()
for name in models.keys():
    plt.plot(fpr_dict[name], tpr_dict[name], label=f"{name} (AUC={auc_dict[name]:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on diabetes10k")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_roc_curves.png"))
plt.close()

# ------------- GRAPH 7: FEATURE IMPORTANCE (RF) -------------
rf = models["RandomForest"]
imps = rf.feature_importances_
idx = np.argsort(imps)
plt.figure(figsize=(6, 8))
plt.barh(X.columns[idx], imps[idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (diabetes10k)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig7_feature_importance.png"))
plt.close()

# ------------- GRAPHS 8–11: CONFUSION MATRICES -------------
for name in models.keys():
    cm = cms[name]
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"fig_cm_{name}.png"))
    plt.close()

print("All graphs saved in folder:", OUT_DIR)
