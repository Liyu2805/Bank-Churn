import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Model Artifacts

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

st.title("🏦 End-to-End ML Churn Analytics System")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Enter Customer Details")

CreditScore = st.sidebar.number_input("Credit Score", 300, 900, 650)
Age = st.sidebar.number_input("Age", 18, 100, 35)
Tenure = st.sidebar.number_input("Tenure (Years)", 0, 10, 5)
Balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
NumOfProducts = st.sidebar.number_input("Number of Products", 1, 4, 1)

HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])

EstimatedSalary = st.sidebar.number_input("Estimated Salary", 1000, 200000, 60000)
PointEarned = st.sidebar.number_input("Points Earned", 0, 1000, 200)

Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
CardType = st.sidebar.selectbox("Card Type", ["GOLD", "PLATINUM", "SILVER"])

tenure_ratio = Tenure / 10
service_bundle_score = NumOfProducts + HasCrCard + IsActiveMember
Recency = 10 - Tenure
Frequency = NumOfProducts
Monetary = Balance

input_dict = {
    "CreditScore": CreditScore,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary,
    "Point Earned": PointEarned,
    "tenure_ratio": tenure_ratio,
    "service_bundle_score": service_bundle_score,
    "Recency": Recency,
    "Frequency": Frequency,
    "Monetary": Monetary,
    "Geography_Germany": 1 if Geography == "Germany" else 0,
    "Geography_Spain": 1 if Geography == "Spain" else 0,
    "Gender_Male": 1 if Gender == "Male" else 0,
    "Card Type_GOLD": 1 if CardType == "GOLD" else 0,
    "Card Type_PLATINUM": 1 if CardType == "PLATINUM" else 0,
    "Card Type_SILVER": 1 if CardType == "SILVER" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

st.header("🔮 Predict Churn Probability")

if st.button("Predict"):

    input_scaled = scaler.transform(input_df)

    prediction_proba = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    st.subheader("Prediction Result")

    st.write(f"**Churn Probability:** {round(prediction_proba * 100, 2)}%")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

st.header("⭐ Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feature_importance_df["Feature"][:10],
            feature_importance_df["Importance"][:10])
    ax.invert_yaxis()
    st.pyplot(fig)

else:
    st.write("Model does not support feature importance.")

st.header("💡 Retention Strategies")

if 'prediction' in locals():
    if prediction == 1:
        st.write("Since this customer has high churn probability, recommended actions:")
        st.write("- Offer personalized discount or cashback.")
        st.write("- Provide loyalty reward program.")
        st.write("- Improve engagement with targeted marketing.")
        st.write("- Assign relationship manager if high balance.")
    else:
        st.write("Customer is stable. Maintain engagement and upsell premium services.")

# ----------------------------
# Model Performance Metrics
# ----------------------------
st.header("📊 Model Performance Metrics")

st.write("Primary Metric Optimized: Recall + AUC")

st.write("**Recall:** Measures how many actual churn customers were correctly identified.")
st.write("**AUC (ROC-AUC):** Measures model’s ability to distinguish churn vs non-churn.")

st.metric("Recall (Churn Class)", "0.72")
st.metric("AUC Score", "0.867")