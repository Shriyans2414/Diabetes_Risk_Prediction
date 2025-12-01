import streamlit as st


#------------------------ DARK MODE CSS ------------------------

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0F0F0F; color: #F0F0F0; }
    /* Sidebar */
    .sidebar .sidebar-content { background-color: #1C1C1C; color: #F0F0F0; }
    /* Headings */
    h1, h2, h3, h4, h5, h6 { color: #00BFFF; }
    /* Tables */
    .dataframe, table { background-color: #1C1C1C; color: #F0F0F0; }
    /* Expander header */
    .streamlit-expanderHeader { color: #00BFFF; }
</style>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import plotly.express as px
import matplotlib as mpl

# ------------------------ DARK MODE PLOTS ------------------------

mpl.rcParams.update({
    'figure.facecolor': '#0F0F0F',
    'axes.facecolor': '#0F0F0F',
    'axes.edgecolor': '#F0F0F0',
    'axes.labelcolor': '#F0F0F0',
    'xtick.color': '#F0F0F0',
    'ytick.color': '#F0F0F0',
    'text.color': '#F0F0F0',
    'grid.color': '#444444',
    'legend.facecolor': '#1C1C1C',
    'legend.edgecolor': '#F0F0F0',
    'lines.linewidth': 2,
    'axes.titleweight': 'bold'
})
# Set dark background for all plots
mpl.rcParams['figure.facecolor'] = '#0F0F0F'      # Figure background
mpl.rcParams['axes.facecolor'] = '#0F0F0F'        # Axes background
mpl.rcParams['axes.edgecolor'] = '#F0F0F0'        # Axes edge color
mpl.rcParams['axes.labelcolor'] = '#F0F0F0'       # X/Y label color
mpl.rcParams['xtick.color'] = '#F0F0F0'           # X ticks
mpl.rcParams['ytick.color'] = '#F0F0F0'           # Y ticks
mpl.rcParams['text.color'] = '#F0F0F0'            # Text color
mpl.rcParams['grid.color'] = '#444444'            # Grid lines
mpl.rcParams['legend.facecolor'] = '#1C1C1C'      # Legend background
mpl.rcParams['legend.edgecolor'] = '#F0F0F0'      # Legend border
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titleweight'] = 'bold'


# ------------------------ PAGE CONFIG ------------------------

st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------ LOAD MODEL ------------------------

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


# ------------------------ TITLE & INTRO (Professional + Subtle Glow) ------------------------

st.markdown("""
<!-- Import Google Font -->
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">

<style>
@keyframes glowPulse {
  0% { text-shadow: 0 0 1px #00FFFF, 0 0 3px #00FFFF; }
  50% { text-shadow: 0 0 2px #00FFFF, 0 0 5px #00FFFF; }
  100% { text-shadow: 0 0 1px #00FFFF, 0 0 3px #00FFFF; }
}
.animated-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 54px;
    color: #00FFFF;
    animation: glowPulse 2s infinite;
    text-align: center;
    font-weight: 700;
}
.animated-subtitle {
    font-family: 'Montserrat', sans-serif;
    font-size: 16px;  /* smaller font */
    color: #CCCCCC;
    text-align: center;
    margin-top: -8px;
    text-shadow: none; /* remove any glow */
}
</style>
<div>
    <div class="animated-title"> DIABETES RISK PREDICTION</div>
    <div class="animated-subtitle">Enter patient details below and see real-time probability of having diabetes.</div>
</div>
""", unsafe_allow_html=True)


# ------------------------ SIDEBAR INPUTS ------------------------

st.sidebar.header(" Input Patient Details")
input_type = st.sidebar.radio("Select input type:", ("Sliders", "Exact Numbers"))

with st.sidebar.expander(" Vitals"):
    preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1) if input_type=="Exact Numbers" else st.slider("Pregnancies", 0, 20, 0)
    age = st.number_input("Age", min_value=0, max_value=120, value=30) if input_type=="Exact Numbers" else st.slider("Age", 0, 120, 30)

with st.sidebar.expander(" Blood Metrics"):
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120) if input_type=="Exact Numbers" else st.slider("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70) if input_type=="Exact Numbers" else st.slider("Blood Pressure", 0, 150, 70)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80) if input_type=="Exact Numbers" else st.slider("Insulin Level", 0, 900, 80)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f") if input_type=="Exact Numbers" else st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)

with st.sidebar.expander(" Body Metrics"):
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f") if input_type=="Exact Numbers" else st.slider("BMI", 0.0, 70.0, 25.0)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20) if input_type=="Exact Numbers" else st.slider("Skin Thickness", 0, 100, 20)

# Health warnings
if glucose > 200:
    st.sidebar.warning("🔴 Very high glucose level! Consider consulting a doctor.")
if bmi > 35:
    st.sidebar.info("🔴 BMI indicates obesity, may increase diabetes risk.")


# ------------------------ LOAD DATA ------------------------

data = pd.read_csv("diabetes10k.csv")  # contains Country column
X = data.drop(["Outcome", "Country"], axis=1)
y = data["Outcome"]



# ------------------------ REAL-TIME PREDICTION ------------------------

df_input = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=X.columns)
df_scaled = scaler.transform(df_input)
prob = model.predict_proba(df_scaled)[0][1] * 100
result = "Diabetes" if prob >= 50 else "No Diabetes"


# ------------------------ DASHBOARD CARDS + CENTERED LIVE GAUGE ------------------------

from streamlit_echarts import st_echarts
st.markdown("---")
#st.markdown("##  Diabetes Risk Dashboard")
# Layout: Gauge (left, centered vertically) + Cards (right)
col1, col2 = st.columns([3, 2], vertical_alignment="center")
with col1:
    st.markdown("<div style='text-align:center; padding-top:130px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFFFFF;'> Live Diabetes Risk Gauge</h3>", unsafe_allow_html=True)
    # Dynamic glow and pulse intensity
    if prob >= 75:
        glow_color = "#E74C3C"
        pulse_strength = 30
    elif prob >= 50:
        glow_color = "#F1C40F"
        pulse_strength = 20
    else:
        glow_color = "#2ECC71"
        pulse_strength = 10
    gauge_options = {
        "animationDuration": 1200,
        "animationEasing": "cubicOut",
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Diabetes Risk",
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": 100,
                "splitNumber": 10,
                "axisLine": {
                    "lineStyle": {
                        "width": 20,
                        "color": [
                            [0.5, "#2ECC71"],
                            [0.75, "#F1C40F"],
                            [1, "#E74C3C"]
                        ],
                        "shadowBlur": pulse_strength,
                        "shadowColor": glow_color,
                    }
                },
                "pointer": {
                    "icon": "path://M2,-5 L-2,-5 L0,10 z",
                    "length": "70%",
                    "width": 6,
                    "itemStyle": {
                        "color": "#FFFFFF",
                        "shadowBlur": pulse_strength,
                        "shadowColor": glow_color
                    },
                },
                "progress": {"show": True, "width": 20, "roundCap": True},
                "detail": {
                    "formatter": "{value}%",
                    "fontSize": 28,
                    "color": glow_color,
                    "offsetCenter": [0, "60%"],
                },
                "data": [{"value": round(prob, 2), "name": "Probability"}],
            }
        ]
    }
    st_echarts(options=gauge_options, height="400px", key="diabetes_gauge")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    # Unified smaller card style
    card_style = """
        background-color:#1C1C1C;
        padding:18px;
        border-radius:15px;
        text-align:center;
        min-height:100px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        color:#F0F0F0;
        box-shadow:0 4px 10px rgba(0,0,0,0.5);
        transition: all 0.6s ease-in-out;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
    """
    # CSS animation for fade-in cards
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
    """, unsafe_allow_html=True)
    # Prediction Card
    st.markdown(
        f"<div style='{card_style}'>"
        f"<h3 style='color:#FFFFFF; margin-bottom:5px;'>Prediction</h3>"
        f"<h2 style='color:#3498DB; margin:0;'>{result}</h2>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    # Probability Card
    st.markdown(
        f"<div style='{card_style}'>"
        f"<h3 style='color:#FFFFFF; margin-bottom:5px;'>Probability</h3>"
        f"<h2 style='color:{glow_color}; margin:0;'>{prob:.2f}%</h2>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    # Risk Message Card
    if prob >= 75:
        message = "🔴 High risk of diabetes. Please consult a doctor."
        color = "#E74C3C"
    elif 50 <= prob < 75:
        message = "🟠 Moderate risk — consider lifestyle changes."
        color = "#F1C40F"
    else:
        message = "🟢 Low risk — maintain a healthy lifestyle!"
        color = "#2ECC71"

    st.markdown(
        f"<div style='{card_style}'>"
        f"<h4 style='color:{color}; margin:0;'>{message}</h4>"
        f"</div>",
        unsafe_allow_html=True
    )
# Subtle pulsing glow for high risk
if prob >= 75:
    st.markdown("""
        <style>
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 10px #E74C3C; }
            50% { box-shadow: 0 0 30px #E74C3C; }
            100% { box-shadow: 0 0 10px #E74C3C; }
        }
        div[data-testid="stHorizontalBlock"] div[role="figure"] {
            animation: pulseGlow 2s infinite ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)


# ------------------------ FEATURE IMPORTANCE ------------------------

st.markdown("---")
st.header(" Model Insights")
with st.expander(" Feature Importance in Prediction "):
    importances = getattr(model, 'feature_importances_', None)
    # Handle ensemble models
    if importances is None and hasattr(model, 'estimators_'):
        importances = model.estimators_[0].feature_importances_
    # Handle linear models (like LogisticRegression)
    if importances is None and hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    # Fallback if no importance attribute exists
    if importances is None:
        importances = [0] * len(X.columns)
    # Normalize importances (so they sum to 1)
    importances = np.array(importances)
    if importances.sum() != 0:
        importances = importances / importances.sum()
    # Build DataFrame
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)
    # Plot
    fig1, ax1 = plt.subplots(figsize=(10,5), facecolor='#0F0F0F')
    sns.barplot(data=importance_df, y="Feature", x="Importance", palette="viridis", ax=ax1)
    ax1.set_facecolor('#0F0F0F')
    # Format labels and style
    ax1.set_xlabel("Relative Importance", color='white', fontsize=12)
    ax1.set_ylabel("")
    ax1.tick_params(colors='white', labelsize=10)
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
    # Show percentage labels on bars
    for i, v in enumerate(importance_df["Importance"]):
        ax1.text(v + 0.005, i, f"{v*100:.1f}%", color='white', va='center')
    st.pyplot(fig1)


# ------------------------ INTERACTIVE PROBABILITY DISTRIBUTION ------------------------

with st.expander("Model Confidence: Probability Distribution (Interactive)", expanded=False):
    st.markdown("""
    This chart shows how confident the model is about predicting diabetes for all patients in the dataset.
    - **Green** → Low probability of diabetes (0-0.3)  
    - **Yellow** → Uncertain / moderate probability (0.3-0.7)  
    - **Red** → High probability of diabetes (0.7-1.0)  
    - **White dashed line** → Decision threshold at 0.5  
    Hover over the bars to see exact counts and probabilities.
    """)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    # Prepare DataFrame for plotting
    results = pd.DataFrame({
        "Actual Outcome": y_test.values,
        "Predicted Probability": probabilities
    })
    # Add risk zones column for better hover info
    def risk_zone(p):
        if p < 0.3:
            return "Low Risk"
        elif p < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
    results["Risk Zone"] = results["Predicted Probability"].apply(risk_zone)
    # Plot histogram
    fig_prob = px.histogram(
        results,
        x="Predicted Probability",
        color="Actual Outcome",
        barmode="overlay",
        nbins=25,
        color_discrete_map={0:"#2ECC71", 1:"#E74C3C"}, 
        opacity=0.7,
        hover_data=["Risk Zone"],
        labels={
            "Actual Outcome": "Outcome",
            "Predicted Probability": "Predicted Probability"
        }
    )
    # Add vertical zones for visual guidance
    fig_prob.add_vrect(x0=0, x1=0.3, fillcolor="#2ECC71", opacity=0.1, line_width=0, annotation_text="Low Risk", annotation_position="top left")
    fig_prob.add_vrect(x0=0.3, x1=0.7, fillcolor="#F1C40F", opacity=0.1, line_width=0, annotation_text="Moderate Risk", annotation_position="top left")
    fig_prob.add_vrect(x0=0.7, x1=1, fillcolor="#E74C3C", opacity=0.1, line_width=0, annotation_text="High Risk", annotation_position="top left")
    # Add decision threshold line
    fig_prob.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="white",
        annotation_text="Threshold (0.5)",
        annotation_position="top left",
        annotation_font_color="white"
    )
    # Layout improvements
    fig_prob.update_layout(
        title="Predicted Probability of Diabetes",
        title_font=dict(size=20, color="white"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white", size=12),
        xaxis=dict(title="Predicted Probability", range=[0,1]),
        yaxis=dict(title="Number of Patients"),
        legend_title_text="Actual Outcome",
        bargap=0.05
    )
    # Make chart interactive
    st.plotly_chart(fig_prob, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Green Zone (0-0.3): Confident patient **does not** have diabetes.  
    - Yellow Zone (0.3-0.7): Model is **uncertain**; retesting recommended.  
    - Red Zone (0.7-1.0): Confident patient **has diabetes**; consult a doctor.  
    - Decision Threshold (0.5): Above this classified as “Diabetes”.
    """)


# ------------------------ INTERACTIVE PIE CHART ------------------------

with st.expander(" Global Diabetes Distribution by Country", expanded=False):
    st.markdown("""
    This interactive visualization shows **diabetes prevalence by country** around the world.  
    Use the dropdown below to filter by region or view all countries combined.  
    Hover over each slice to see the **percentage share** and **average prevalence**.
    """)
    import plotly.express as px
    # --- Map countries to regions ---
    region_map = {
        "India": "Asia", "China": "Asia", "Japan": "Asia", "South Korea": "Asia", "Indonesia": "Asia",
        "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe", "Italy": "Europe", "Spain": "Europe",
        "United States": "North America", "Canada": "North America", "Mexico": "North America",
        "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
        "South Africa": "Africa", "Nigeria": "Africa", "Egypt": "Africa", "Kenya": "Africa",
        "Australia": "Oceania", "New Zealand": "Oceania"
    }
    # Add region column
    data["Region"] = data["Country"].map(region_map).fillna("Other")
    # --- Dropdown filter ---
    regions = sorted(data["Region"].unique())
    selected_region = st.selectbox(" Select Region:", ["All"] + regions)
    filtered_data = (
        data if selected_region == "All"
        else data[data["Region"] == selected_region]
    )
    # --- Compute diabetes prevalence per country ---
    country_stats = (
        filtered_data.groupby("Country")["Outcome"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    # --- Handle empty region selection ---
    if country_stats.empty:
        st.warning("No data available for this region.")
    else:
        # --- Custom color palette ---
        color_palette = (
            px.colors.qualitative.Vivid
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Bold
            + px.colors.qualitative.Pastel
        )
        # --- Create Pie Chart ---
        fig = px.pie(
            country_stats,
            names="Country",
            values="Outcome",
            color="Country",
            color_discrete_sequence=color_palette,
            hole=0.35,
            title=f"🩺 Diabetes Prevalence — {selected_region if selected_region != 'All' else 'Global'}",
        )
        # --- Layout and style ---
        fig.update_traces(
            textinfo="percent+label",
            textfont_size=14,
            pull=[0.04] * len(country_stats),
            hovertemplate="<b>%{label}</b><br>Prevalence: %{value:.2f}<br>Share: %{percent}<extra></extra>"
        )
        fig.update_layout(
            showlegend=True,
            legend_title_text="Country",
            title_font=dict(size=22, family="Helvetica", color="white"),
            paper_bgcolor="#0F0F0F",
            plot_bgcolor="#0F0F0F",
            font=dict(color="white", size=13),
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.3,
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    This chart helps identify which countries in your dataset have the highest diabetes prevalence,  
    guiding targeted awareness or health initiatives.
    """)


# ------------------------ STATISTICAL INSIGHTS ------------------------
st.markdown("---")
st.header("Statistical Insights in Diabetes Prediction")
with st.expander(" Data Understanding (Descriptive Statistics)"):
    st.markdown("""
    We start by analyzing the dataset using descriptive statistics to understand patterns.
    **Examples:**
    - Mean glucose level shows average sugar in the dataset  
    - BMI variance shows how body weight differs among patients  
    - Correlation helps identify strong predictors of diabetes
    """)
    st.write(" **Feature Correlation Matrix:**")
    corr = data.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
with st.expander(" Model Building (Inferential Statistics)"):
    st.markdown("""
    Your model is built using **probability and statistical inference**.
    - Logistic regression (or Random Forest) estimates how each factor affects diabetes likelihood.  
    - Coefficients or feature importances tell us how much each input contributes to the prediction.  
    - Statistical estimation methods like *Maximum Likelihood* find best-fitting parameters.
    """)
    if hasattr(model, 'coef_'):
        st.write("**Logistic Regression Coefficients:**")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", ascending=False)
        st.dataframe(coef_df)
    else:
        st.write("**Feature Importance (Statistical Influence):**")
        importances_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importances_df)
with st.expander(" Model Evaluation (Statistical Testing)"):
    st.markdown("""
    We use statistical measures to check performance:
    - **Accuracy** → Proportion of correct predictions  
    - **Precision & Recall** → Derived from conditional probabilities  
    - **AUC-ROC Curve** → Statistical separability between diabetic and non-diabetic  
    - **Cross-validation** → Ensures model stability
    """)
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    y_pred = model.predict(scaler.transform(X))
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, model.predict_proba(scaler.transform(X))[:,1])
    st.write(f"**Model Accuracy:** {accuracy*100:.2f}%")
    st.write(f"**AUC Score:** {auc:.2f}")
    fpr, tpr, _ = roc_curve(y, model.predict_proba(scaler.transform(X))[:,1])
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax_roc.plot([0,1],[0,1],'--',color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (Statistical Model Evaluation)")
    ax_roc.legend()
    st.pyplot(fig_roc)
with st.expander(" Summary of Statistical Approach"):
    st.markdown("""
    | Stage | Statistical Concept | Purpose |
    |:------|:--------------------|:----------|
    | Data Cleaning | Mean/Median Imputation | Handle missing values |
    | Feature Analysis | Correlation, Hypothesis Testing | Identify key predictors |
    | Model Training | Logistic Regression / Probability | Build predictive model |
    | Evaluation | Accuracy, AUC, Confidence Intervals | Test model reliability |

    >  **Statistics explains why your model works — Machine Learning shows how well it works.**
    """)
# --- Prepare Report Data ---
report_df = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [glucose],
    "Blood Pressure": [bp],
    "Skin Thickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "Diabetes Pedigree Function": [dpf],
    "Age": [age],
    "Predicted Probability (%)": [prob],
    "Prediction": [result],
    "Risk Message": [message]
})


# ------------------------ DETAILED STATISTICAL BREAKDOWN ------------------------

st.markdown("---")
st.header(" Statistical Breakdown for the Given Input")
with st.expander(" Detailed Statistical Interpretation for Your Input"):
    st.markdown("""
    This section explains **how statistics help interpret your entered values** and how they influence
    the diabetes probability prediction.
    """)
    # Mean and Std (based on Pima dataset averages)
    stats_data = {
    "Feature": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "Mean": [3.8, 121.7, 72.4, 29.1, 155.5, 32.0, 0.47, 33.2],
    "StdDev": [3.4, 30.5, 12.4, 9.6, 118.8, 7.9, 0.33, 11.8],
    "InputValue": [preg, glucose, bp, skin, insulin, bmi, dpf, age]
}
    df_stats = pd.DataFrame(stats_data)
# Round float features
    float_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
    df_stats.loc[df_stats["Feature"].isin(float_features), ["Mean", "StdDev", "InputValue"]] = \
    df_stats.loc[df_stats["Feature"].isin(float_features), ["Mean", "StdDev", "InputValue"]].round(2)
# Convert integer-like features to int
    int_features = ["Pregnancies", "Age"]
    df_stats.loc[df_stats["Feature"].isin(int_features), ["Mean", "StdDev", "InputValue"]] = \
     df_stats.loc[df_stats["Feature"].isin(int_features), ["Mean", "StdDev", "InputValue"]].astype(int)
# Calculate Z-Score
    df_stats["Z-Score"] = ((df_stats["InputValue"] - df_stats["Mean"]) / df_stats["StdDev"]).round(2)
# Define gradient coloring for Z-Score: green for negative, red for positive
    def color_z(val):
     if val > 0:
        color = f"rgba(255,0,0,{min(val/3,1)})"  # Red shade
     elif val < 0:
        color = f"rgba(0,128,0,{min(abs(val)/3,1)})"  # Green shade
     else:
        color = ""
     return f"background-color: {color}"
# Display table
    st.write("### Statistical Profile and Z-Scores")
    st.dataframe(
     df_stats.style
     .format({
        "Mean": "{:.2f}",
        "StdDev": "{:.2f}",
        "InputValue": lambda x: f"{x:.2f}" if isinstance(x, float) else f"{int(x)}",
        "Z-Score": "{:.2f}"
    })
    .applymap(color_z, subset=["Z-Score"])
)
# Explanation
    st.markdown("""
- **Z-Score** tells how far your value is from the average of the population (in standard deviations).  
- Positive Z-Scores (red) indicate your value is above average.  
- Negative Z-Scores (green) indicate your value is below average.  
- Example: A Z-score of **+1.23** means your value is 1.23 std above the mean — higher than ~89% of others.  
""")
    # Simple textual interpretation
    def interpret_z(z):
        if z >= 1.0: return "🚨 High (above average)"
        elif z >= 0.5: return "⚠️ Slightly above average"
        elif z <= -1.0: return "🟢 Low (below average)"
        elif z <= -0.5: return "🟢 Slightly below average"
        else: return "⚪ Normal range"

    df_stats["Interpretation"] = df_stats["Z-Score"].apply(interpret_z)

    st.write("###  Statistical Interpretation per Feature")
    st.dataframe(df_stats[["Feature", "Z-Score", "Interpretation"]])


# ------------------------ INTERACTIVE Z-SCORE RADAR CHART ------------------------

import plotly.graph_objects as go
with st.expander(" Interactive Z-Score Dashboard", expanded=False):
    st.markdown("""
    This interactive chart shows **Z-Score deviations** per feature:
    - **Red bars** → Above average (higher risk)
    - **Green bars** → Below average (safer)
    - Hover over each bar for exact Z-score and interpretation.
    """)
    categories = df_stats["Feature"].tolist()
    z_scores = df_stats["Z-Score"].tolist()
    interpretations = df_stats["Interpretation"].tolist()
    fig = go.Figure()
    for i, feature in enumerate(categories):
        r = abs(z_scores[i])
        color = "#E74C3C" if z_scores[i] > 0 else "#2ECC71"
        fig.add_trace(go.Barpolar(
            r=[r],
            theta=[i * (360/len(categories))],
            width=[360/len(categories)-5],
            marker_color=color,
            marker_line_color="white",
            marker_line_width=2,
            opacity=0.85,
            name=feature,
            hovertemplate=f"<b>{feature}</b><br>Z-Score: {z_scores[i]}<br>{interpretations[i]}<extra></extra>"
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='black',
            radialaxis=dict(range=[0, max(abs(z) for z in z_scores)+0.5],
                            tickfont=dict(color='white'),
                            gridcolor='#444444'),
            angularaxis=dict(
                tickmode='array',
                tickvals=[i*(360/len(categories)) for i in range(len(categories))],
                ticktext=categories,
                tickfont=dict(color='white')
            )
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        title=dict(text=" Z-Score Radar Dashboard", font=dict(size=22, color='white')),
        legend=dict(font=dict(color='white')),
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)
# Explanation text
    st.markdown("""
### What the Model Learns from These Statistics
- Each feature’s deviation (Z-score) helps estimate its **statistical weight**.
- Features like **Glucose**, **Age**, and **BMI** usually have the strongest predictive influence.
- Logistic Regression converts these weighted contributions into probabilities using:
""")
    st.latex(r"P(\text{Diabetes}) = \frac{1}{1 + e^{-(b_0 + b_1 X_1 + b_2 X_2 + \dots + b_8 X_8)}}")
    st.markdown("""
**Explanation of the Formula:**

- `P(Diabetes)` → Probability that the patient has diabetes (ranges from 0 to 1).  
- `b_0` → Intercept term (baseline risk when all features are zero).  
- `b_1 ... b_8` → Coefficients learned by the model for each feature.  
- `X_1 ... X_8` → Patient’s feature values (e.g., Glucose, BMI, Age, etc.).  
- `e` → Euler's number, used to map the weighted sum into a probability between 0 and 1.  

**How it works:**  
1. Each feature is multiplied by its coefficient to reflect its influence on diabetes risk.  
2. All contributions are summed with the intercept (`b_0`).  
3. The negative of this sum is exponentiated and added to 1.  
4. Finally, the inverse (`1 / (...)`) gives a probability between 0 and 1.  

> In simple terms: **higher weighted sums → higher probability of diabetes**, lower sums → lower probability.  
""")
    st.markdown("""
- Decision Trees or Random Forests use **entropy reduction** to statistically split data at points that reduce uncertainty most.
### Summary
| Statistical Concept | Purpose |
|:--|:--|
| Mean & Std Dev | Measure central tendency & variation |
| Z-Score | Show how unusual your input is |
| Correlation | Find which features relate to diabetes |
| Probability Model | Converts feature effects into risk probability |
| Entropy / Gini | Measures information gain in classification |
>  *Statistics explains why your model predicts the way it does.*
""")
    st.success("This explanation is based on your exact input — helping you understand your risk in a transparent, data-driven way.")

# ------------------------ LIVE INTERACTIVE MODEL PERFORMANCE DASHBOARD ------------------------

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ------------------------ PAGE HEADER ------------------------
st.markdown("---")
st.markdown("<h2 style='text-align:center; color:#00FFFF;'>Live Interactive Model Performance Dashboard</h2>", unsafe_allow_html=True)

# ------------------------ LOAD DATA ------------------------
data = pd.read_csv("diabetes10k.csv")  # Replace with your dataset path

# ------------------------ SELECT TOP FEATURES ------------------------
top_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction", "Insulin", "SkinThickness"]
X = data[top_features]
y = data["Outcome"]

# ------------------------ TRAIN-TEST SPLIT ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------ SCALE FEATURES ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ DEFINE MODELS ------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# ------------------------ TRAIN MODELS & COLLECT METRICS ------------------------
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_test, y_pred, zero_division=0) * 100
    }

# ------------------------ DISPLAY ALL MODELS METRICS ------------------------
st.subheader("All Models Metrics")
metrics_df = pd.DataFrame(results).T  # Transpose so models are rows
metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1"]]  # Ensure column order
st.dataframe(metrics_df.style.format("{:.2f}"))

# ------------------------ SELECT MODEL TO VIEW ------------------------
selected_model = st.selectbox("Select Model to View Metrics", list(results.keys()))
metrics = results[selected_model]

st.markdown(f"""
<div style='background-color:#0F0F0F; padding:20px; border-radius:15px; border: 2px solid #00FFFF; text-align:center; margin-top:20px;'>
    <h3 style='color:#00FFFF;'> Model: {selected_model}</h3>
    <p style='color:white; font-size:16px;'>
        Accuracy: <b style='color:#1ABC9C;'>{metrics["Accuracy"]:.2f}%</b> &nbsp; | &nbsp;
        Precision: <b style='color:#F39C12;'>{metrics["Precision"]:.2f}%</b> &nbsp; | &nbsp;
        Recall: <b style='color:#E74C3C;'>{metrics["Recall"]:.2f}%</b> &nbsp; | &nbsp;
        F1: <b style='color:#9B59B6;'>{metrics["F1"]:.2f}%</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------ DISPLAY BEST MODEL ------------------------
top_model = max(results, key=lambda m: results[m]["Accuracy"])
top_metrics = results[top_model]

st.markdown(f"""
<div style='background-color:#0F0F0F; padding:25px; border-radius:15px; border: 3px solid #00FFFF; text-align:center; margin-top:20px;'>
    <h2 style='color:#00FFFF;'> Best Performing Model: {top_model}</h2>
    <p style='color:white; font-size:16px;'>
        Accuracy: <b style='color:#1ABC9C;'>{top_metrics["Accuracy"]:.2f}%</b> &nbsp; | &nbsp;
        Precision: <b style='color:#F39C12;'>{top_metrics["Precision"]:.2f}%</b> &nbsp; | &nbsp;
        Recall: <b style='color:#E74C3C;'>{top_metrics["Recall"]:.2f}%</b> &nbsp; | &nbsp;
        F1: <b style='color:#9B59B6;'>{top_metrics["F1"]:.2f}%</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------ TOOLTIP DATA ------------------------

tooltip_data = [
    f"<b>{m}</b><br>Accuracy: {results[m]['Accuracy']:.2f}%<br>Precision: {results[m]['Precision']:.2f}%<br>Recall: {results[m]['Recall']:.2f}%<br>F1 Score: {results[m]['F1']:.2f}%"
    for m in results
]
st.markdown("<br><br>", unsafe_allow_html=True)


# ------------------------ PREPARE METRIC ARRAYS FOR CHART ------------------------
model_names = list(results.keys())
acc = [results[m]["Accuracy"] for m in model_names]
prec = [results[m]["Precision"] for m in model_names]
rec = [results[m]["Recall"] for m in model_names]
f1_scores = [results[m]["F1"] for m in model_names]


# ------------------------ LIVE DASHBOARD EXPANDER ------------------------

with st.expander(" Live Model Performance Dashboard", expanded=True):
    # ------------------------ METRIC TOGGLES ------------------------
    st.markdown("### Select Metrics to Display")
    # Create 4 equal columns for horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    show_acc = col1.checkbox("Accuracy", value=True)
    show_prec = col2.checkbox("Precision", value=True)
    show_rec = col3.checkbox("Recall", value=True)
    show_f1 = col4.checkbox("F1 Score", value=True)
    # ------------------------ CREATE SERIES BASED ON TOGGLES ------------------------
    series = []
    if show_acc:
        series.append({
            "name": "Accuracy",
            "type": "bar",
            "data": [
                {
                    "value": acc[i],
                    "itemStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "#00FFFF"},
                                {"offset": 1, "color": "#0066FF"}
                            ]
                        },
                        "shadowBlur": 15,
                        "shadowColor": "#00FFFF"
                    }
                } for i in range(len(list(results.keys())))
            ],
            "barWidth": "25%",
            "label": {"show": True, "position": "top", "color": "#FFFFFF", "formatter": "{c}%"},
        })
    if show_prec:
        series.append({
            "name": "Precision",
            "type": "line",
            "data": prec,
            "smooth": True,
            "lineStyle": {"color": "#F39C12", "width": 3},
            "itemStyle": {"color": "#F39C12"},
            "symbolSize": 9,
            "emphasis": {"scale": True}
        })
    if show_rec:
        series.append({
            "name": "Recall",
            "type": "line",
            "data": rec,
            "smooth": True,
            "lineStyle": {"color": "#E74C3C", "width": 3},
            "itemStyle": {"color": "#E74C3C"},
            "symbolSize": 9,
            "emphasis": {"scale": True}
        })
    if show_f1:
        series.append({
            "name": "F1 Score",
            "type": "line",
            "data": f1_scores,
            "smooth": True,
            "lineStyle": {"color": "#9B59B6", "width": 3},
            "itemStyle": {"color": "#9B59B6"},
            "symbolSize": 9,
            "emphasis": {"scale": True}
        })
    # ------------------------ ECHARTS OPTIONS ------------------------
    chart_options = {
        "backgroundColor": "#0F0F0F",
        "animationDuration": 1800,
        "animationEasing": "elasticOut",
        "title": {
            "text": " Model Comparison — Performance Metrics",
            "left": "center",
            "textStyle": {"color": "#1ABC9C", "fontSize": 18}
        },
        "tooltip": {"show": False},  # Tooltip disabled
        "legend": {
            "data": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "top": "8%",
            "textStyle": {"color": "#FFFFFF", "fontSize": 12}
        },
        "xAxis": {
            "type": "category",
            "data": model_names,
            "axisLabel": {"color": "#FFFFFF", "rotate": 10},
            "axisLine": {"lineStyle": {"color": "#AAAAAA"}}
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 100,
            "axisLabel": {"color": "#FFFFFF"},
            "axisLine": {"lineStyle": {"color": "#AAAAAA"}},
            "splitLine": {"lineStyle": {"color": "#333333"}}
        },
        "series": series
    }
    # ------------------------ DISPLAY CHART ------------------------
    st_echarts(options=chart_options, height="520px")


# ------------------------ DOWNLOAD REPORT ------------------------

st.markdown("---")
st.subheader("Download Your Prediction Report")


# ------------------------ DOWNLOAD REPORT ------------------------

report_df = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [glucose],
    "Blood Pressure": [bp],
    "Skin Thickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "Diabetes Pedigree Function": [dpf],
    "Age": [age],
    "Predicted Probability (%)": [prob],
    "Prediction": [result],
    "Risk Message": [message]
})
csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV", data=csv, file_name='diabetes_prediction_report.csv', mime='text/csv')
st.markdown("---")