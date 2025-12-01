import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("diabetes10k.csv")
X = data.drop(["Outcome", "Country"], axis=1)
y = data["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                    use_label_encoder=False, eval_metric='logloss', random_state=42)

# Voting ensemble
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

cv_scores = cross_val_score(voting_clf, X_scaled, y, cv=5)
print("5-Fold CV Accuracy:", cv_scores.mean())

# Save model
joblib.dump(voting_clf, "diabetes_model.pkl")
print("✅ Model and scaler saved successfully!")
