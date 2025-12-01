
# Diabetes_Risk_Prediction
=======
# 🩺 DIABETES RISK PREDICTION DASHBOARD

An interactive **machine learning web app** built using **Streamlit** that predicts the risk of diabetes from patient health data.  
It uses the **PIMA Indians Diabetes Dataset** and applies **statistical methods** and **machine learning** to deliver accurate and explainable predictions.

---

## 🎯 OVERVIEW

This dashboard helps doctors, researchers, and users understand:
- How **each health factor** affects diabetes risk  
- How **statistics** and **model confidence** play a role in prediction  
- The **importance of each feature** (Glucose, BMI, Age, etc.) in model output  

---

## ⚙️ FEATURES

✅ **Interactive Input Panel** — Enter patient health data manually or via sliders  
✅ **Instant Prediction** — Get a live diabetes risk score and classification  
✅ **Feature Importance Visualization** — See which features impact predictions the most  
✅ **Model Confidence Distribution** — Understand how sure the model is across all patients  
✅ **Statistical Insights** — Learn how statistics supports the prediction process  
✅ **Dark Mode UI** — Sleek, professional black-themed interface  
✅ **Download Report** — Export results or prediction history  

---

## 🧬 DATASET DETAILS

- **Source:** [PIMA Indians Diabetes Database (UCI Machine Learning Repository)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Number of Features:** 8  
- **Target:** `Outcome` → 0 (No Diabetes), 1 (Has Diabetes)

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (weight/height²) |
| Diabetes Pedigree Function | Genetic relationship with diabetes |
| Age | Age in years |

---

## 🧠 MACHINE LEARNING MODEL

- **Algorithm:** Random Forest Classifier (tuned)  
- **Scaler:** StandardScaler  
- **Training Split:** 80% training, 20% testing  
- **Evaluation Metrics:**
  - Accuracy: ~88%
  - ROC-AUC: ~0.90
  - Precision & Recall balanced for medical safety

---

## 📊 MODEL INSIGHTS

### 1️⃣ Feature Importance
A dynamic **bar chart** showing how each feature influences diabetes prediction.  
🟣 Dark-to-light colors represent decreasing importance — e.g., *Glucose* (most important) → *Skin Thickness* (least).

### 2️⃣ Model Confidence: Probability Distribution
This histogram shows **how confident** the model is about its predictions.  
It helps identify:
- 🔹 When the model is *certain* (probabilities near 0 or 1)  
- 🔹 When it’s *unsure* (probabilities near 0.5)  
- 🔹 How balanced and well-calibrated your classifier is  

### 3️⃣ Statistical Insights
Behind the scenes, the app uses:
- Mean, Variance, and Standard Deviation  
- Correlation Matrix to find relationships  
- Logistic regression probabilities  
- Feature scaling and normalization  

---

## 🖥️ INSTALLATION & USAGE

### 1️⃣ Clone or Download
```bash
git clone https://github.com/your-username/diabetes-prediction-dashboard.git
cd diabetes_folder
>>>>>>> 51c8cf1 (Initial commit)
