# ============================================
# app.py - Churn Risk Predictor
# Run: streamlit run app.py
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="Churn Predictor", page_icon="phone")
st.title("Customer Churn Risk Predictor")
st.write("Enter customer details to predict churn probability")

# Load artifacts
model = joblib.load('xgb_model.joblib')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

col1, col2, col3 = st.columns(3)

tenure = col1.slider("Tenure (months)", 0, 72, 12)
monthly = col2.number_input("Monthly Charges", 18.0, 120.0, 70.0)
total = col3.number_input("Total Charges", 0.0, 9000.0, float(monthly * tenure))

contract = st.selectbox("Contract Type",
                        ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method",
                       ["Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check"])
senior = st.selectbox("Senior Citizen?", ["No", "Yes"])

if st.button("Predict Churn Risk"):
    # Build input vector in exact training column order
    vec = pd.DataFrame(np.zeros((1, len(feature_names))),
                       columns=feature_names)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}

    vec['tenure'] = tenure
    vec['MonthlyCharges'] = monthly
    vec['TotalCharges'] = total
    vec['Contract'] = contract_map[contract]
    vec['InternetService'] = internet_map[internet]
    vec['SeniorCitizen'] = 1 if senior == "Yes" else 0

    if payment == "Electronic check":
        vec['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check":
        vec['PaymentMethod_Mailed check'] = 1
    elif payment == "Credit card (automatic)":
        vec['PaymentMethod_Credit card (automatic)'] = 1
    # Bank transfer = all dummy columns 0 (drop_first baseline)

    # Scale numeric columns with saved scaler
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    vec[num_cols] = scaler.transform(vec[num_cols])

    proba = model.predict_proba(vec)[0][1]

    st.metric("Churn Probability", f"{proba:.1%}")
    if proba > 0.7:
        st.subheader("HIGH RISK")
        st.error("Recommendation: Offer annual contract conversion "
                 "+ free TechSupport add-on immediately.")
    elif proba > 0.4:
        st.subheader("MEDIUM RISK")
        st.warning("Recommendation: Enroll in autopay migration campaign; "
                   "monitor usage.")
    else:
        st.subheader("LOW RISK")
        st.success("Customer healthy. Maintain current engagement.")

st.write("---")
st.caption("Built with XGBoost + SHAP | AUC ~0.85 on test set")
