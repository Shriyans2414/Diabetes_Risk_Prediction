# Diabetes Risk Prediction Dashboard

This project is an interactive machine learning web application built using Streamlit to predict the risk of diabetes based on patient health data. The system uses the PIMA Indians Diabetes Dataset and incorporates statistical methods and machine learning techniques to provide accurate and explainable predictions.

---

## Overview

The dashboard allows users to:
- Understand how each health factor affects diabetes risk
- View statistical insights to interpret predictions
- Explore model confidence and feature importance
- Interact with a clean, responsive interface for making predictions

---

## Features

- Interactive input panel for patient data
- Instant prediction with probability scores
- Feature importance visualization
- Model confidence distribution
- Statistical insights (correlation, variance, scaling effects)
- Dark mode interface
- Downloadable prediction report

---

## Dataset Details

**Source:**  
PIMA Indians Diabetes Database (UCI / Kaggle)  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Features (8):**

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body Mass Index |
| Diabetes Pedigree Function | Genetic relationship with diabetes |
| Age | Age in years |

**Target Variable:**  
`Outcome` → 0 (No Diabetes), 1 (Diabetes)

---

## Machine Learning Model

- Algorithm: Random Forest Classifier  
- Preprocessing: StandardScaler  
- Train/Test Split: 80/20  
- Evaluation Metrics:  
  - Accuracy: ~88%  
  - ROC-AUC: ~0.90  
  - Balanced precision and recall

---

### Disclaimer

Not a medical device. This project is for educational and research purposes only. It is not intended for clinical diagnosis or treatment. Always consult a qualified healthcare professional for medical decisions.

---

### Installation and Usage

```bash
# Clone the Repository
git clone https://github.com/Shriyans2414/Diabetes_Risk_Prediction.git
cd Diabetes_Risk_Prediction

# Install Dependencies
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the Application
streamlit run diabetes_app.py




