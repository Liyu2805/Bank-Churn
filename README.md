🏦 End-to-End ML Churn Analytics System

📌 Project Overview
This project is an end-to-end Machine Learning system built to predict customer churn in a banking dataset.

It includes:
Data preprocessing & feature engineering
Model training using XGBoost
Performance evaluation using Recall & ROC-AUC
Deployment using Streamlit
Interactive dashboard with real-time churn prediction
Feature importance visualization
Retention strategy suggestions
The system helps identify customers likely to churn and supports proactive retention planning.

🎯 Business Objective
Customer churn directly impacts revenue.
The goal of this system is to:
Accurately identify high-risk customers
Optimize recall to capture maximum churners
Provide actionable retention strategies
Enable business teams to make data-driven decisions

🧠 Model Details
Algorithm: XGBoost Classifier
Primary Metric Optimized: Recall + ROC-AUC
AUC Score: 0.867
Recall (Churn Class): 0.72
Accuracy: 0.81
Why Recall?
In churn prediction, missing a churner is more costly than falsely flagging a stable customer.
Therefore, Recall was prioritized to capture maximum at-risk customers.
🛠 Feature Engineering
Custom features created:
tenure_ratio
service_bundle_score
RFM Features (Recency, Frequency, Monetary)
Categorical features encoded using one-hot encoding.

📊 Dashboard Features
The Streamlit dashboard includes:
🔮 Predict Churn Probability
Takes customer inputs
Displays churn probability
Shows final classification (Churn / Not Churn)
⭐ Feature Importance
Displays top contributing features
Helps interpret model decisions
💡 Retention Strategies
Suggests actions for high-risk customers
Business-ready insights

📊 Model Performance Metrics
Recall (Churn Class)
ROC-AUC Score
🚀 Live Deployment
Deployed using Streamlit Cloud.
💻 Installation Guide (Run Locally)
Follow these steps to run the project on your system:
1️⃣ Clone the Repository
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create Virtual Environment (Recommended)
Bash
python -m venv venv

Activate it:
Windows:

Bash
venv\Scripts\activate

Mac/Linux:

Bash
source venv/bin/activate

3️⃣ Install Dependencies
If you have requirements.txt:
Copy code
Bash
pip install -r requirements.txt
If not, install manually:
Copy code
Bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib joblib
4️⃣ Ensure Model Files Are Present
Make sure these files exist in the root directory:
churn_model.pkl
scaler.pkl
feature_columns.pkl
5️⃣ Run the Streamlit App

Bash
streamlit run app.py
Your browser will open automatically.

📁 Project Structure
Copy code

├── app.py
├── churn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md

👨‍💻 Author - Liyutsa Zirange 
Developed as part of ML Internship Project.
