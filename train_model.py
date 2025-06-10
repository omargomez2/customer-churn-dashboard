import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and clean the data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Encode target
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# One-hot encode 'Contract' column
df = pd.get_dummies(df, columns=["Contract"], drop_first=False)

# Select features
features = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year"
]
X = df[features]
y = df["Churn"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and features
joblib.dump((model, features), "churn_model.pkl")

print("✅ Model trained and saved as churn_model.pkl")
