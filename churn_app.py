import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and feature order
model, features = joblib.load("churn_model.pkl")

# Load original data for EDA
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("This app allows you to explore customer churn data and make predictions.")

# --- 1. Overall Churn Distribution ---
st.subheader("📌 Overall Churn Distribution")
fig1, ax1 = plt.subplots()
df['Churn'].value_counts().plot(kind="bar", color=["green", "red"], ax=ax1)
ax1.set_title("Churn Count")
ax1.set_ylabel("Customers")
st.pyplot(fig1)

# --- 2. Churn Rate by Contract Type ---
st.subheader("📉 Churn Rate by Contract Type")
contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0)
if 'Yes' in contract_churn.columns:
    fig2, ax2 = plt.subplots()
    contract_churn['Yes'].plot(kind="bar", color="salmon", ax=ax2)
    ax2.set_ylabel("Churn Rate")
    ax2.set_title("Churn Rate by Contract Type")
    st.pyplot(fig2)
else:
    st.warning("Could not find 'Yes' churn values. Please check the dataset.")

# --- 3. Prediction ---
st.subheader("🔮 Predict Customer Churn")
st.markdown("Enter the customer data below:")

# Sample input fields
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

# One-hot encode contract type
contract_encoded = {
    'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
}

# Build input sample as DataFrame
sample_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    **contract_encoded
}
sample = pd.DataFrame([sample_dict])

# Reorder columns to match model training
sample = sample[features]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    st.markdown(f"### ✅ Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.markdown(f"### 📈 Probability of churn: {prob:.2%}")


