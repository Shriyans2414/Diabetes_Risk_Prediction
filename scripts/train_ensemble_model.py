# train_ensemble_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
data = pd.read_csv("diabetes10k.csv")

# Step 2: Separate features and target
X = data.drop(["Outcome", "Country"], axis=1)
y = data["Outcome"]

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=150, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)

# Step 6: Combine models (Voting Ensemble)
ensemble_model = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('gb', gb)],
    voting='soft'  # soft = use predicted probabilities
)

# Step 7: Train ensemble
ensemble_model.fit(X_train_scaled, y_train)

# Step 8: Evaluate
y_pred = ensemble_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Save model and scaler
joblib.dump(ensemble_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Ensemble model and scaler saved successfully!")
