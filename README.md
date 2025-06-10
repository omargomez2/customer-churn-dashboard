# 📊 Customer Churn Prediction Dashboard

An interactive Streamlit application that visualizes and predicts customer churn using a Random Forest model trained on the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn). This dashboard includes visual insights and allows users to simulate churn predictions based on sample data.

---

## 🚀 Features

- Interactive visualizations of churn by contract type and tenure
- Predict whether a customer will churn based on input features
- Streamlit-powered interface for easy use
- Simple training script to build your own churn model

---

## 📂 Project Structure

customer-churn-dashboard/
│
├── churn.csv # Dataset (from Kaggle)
├── churn_app.py # Streamlit app
├── train_model.py # Model training script
├── churn_model.pkl # Trained model (generated)
└── README.md # Project documentation

streamlit run churn_app.py
