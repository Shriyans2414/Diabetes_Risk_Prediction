# visualize_model.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and data
model = joblib.load("diabetes_model.pkl")
data = pd.read_csv("diabetes10k.csv")
X = data.drop(["Outcome", "Country"], axis=1)


# Some ensemble models (like GradientBoosting or RandomForest)
# have feature_importances_ attribute
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model.estimators_[1][1], 'feature_importances_'):
    # For VotingClassifier, get from RandomForest (index 1)
    importances = model.estimators_[1][1].feature_importances_
else:
    raise ValueError("Model does not have feature importances")

# Plot
plt.figure(figsize=(8, 5))
plt.barh(X.columns, importances, color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Diabetes Prediction")
plt.gca().invert_yaxis()
plt.show()

# Enhanced Probability Distribution Visualization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data again
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale test data
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# Get predicted probabilities
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Create a dataframe for plotting
results = pd.DataFrame({
    "Actual Outcome": y_test.values,
    "Predicted Probability": probabilities
})

# Plot distributions for both classes
plt.figure(figsize=(9, 5))
sns.histplot(data=results, x="Predicted Probability", hue="Actual Outcome",
             bins=20, kde=True, palette=["green", "red"], alpha=0.5)

plt.title("🧠 Model Confidence: Probability Distribution by Actual Outcome")
plt.xlabel("Predicted Probability of Diabetes")
plt.ylabel("Number of Patients")
plt.legend(["No Diabetes (0)", "Diabetes (1)"])
plt.show()
