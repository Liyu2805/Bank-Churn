import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Model Artifacts

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("End-to-end ML Churn Analytics System")

st.sidebar.header("Customer Information")

CreditScore = st.sidebar.number_input("Credit Score", 300, 900, 650)
Age = st.sidebar.number_input("Age", 18, 100, 30)
Tenure = st.sidebar.number_input("Tenure (Years)", 0, 10, 3)
Balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
NumOfProducts = st.sidebar.number_input("Number of Products", 1, 4, 1)

HasCrCard = st.sidebar.selectbox("Has Credit Card", [0,1])
IsActiveMember = st.sidebar.selectbox("Active Member", [0,1])

EstimatedSalary = st.sidebar.number_input("Estimated Salary", 1000, 200000, 50000)
PointEarned = st.sidebar.number_input("Points Earned", 0, 1000, 200)

Geography = st.sidebar.selectbox("Geography", ["France","Germany","Spain"])
Gender = st.sidebar.selectbox("Gender", ["Female","Male"])
CardType = st.sidebar.selectbox("Card Type", ["GOLD","PLATINUM","SILVER"])

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

"Geography_Germany": 1 if Geography=="Germany" else 0,
"Geography_Spain": 1 if Geography=="Spain" else 0,

"Gender_Male": 1 if Gender=="Male" else 0,

"Card Type_GOLD": 1 if CardType=="GOLD" else 0,
"Card Type_PLATINUM": 1 if CardType=="PLATINUM" else 0,
"Card Type_SILVER": 1 if CardType=="SILVER" else 0
}

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Align columns
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

if st.button("🔮 Predict Churn Probability"):

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prob*100:.2f}%"
    )

    # Risk Level + Recommendations
    if prob > 0.5:
        st.error("⚠ High Risk Customer")

        st.markdown("""
        ### Retention Suggestions:
        - Offer loyalty rewards  
        - Provide discount offers  
        - Improve engagement programs  
        """)

    else:
        st.success("✅ Low Risk Customer")

st.header("⭐ Feature Importance")

if st.button("Show Feature Importance"):

    importance = model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(feat_imp["Feature"], feat_imp["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)


st.write("""
- XGBoost Model Used  
- Trained on Bank Customer Churn Dataset  
- Primary metric optimized = Recall + AUC  
""")