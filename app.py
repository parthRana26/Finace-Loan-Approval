import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('loan_approval_pipeline.pkl')

st.title("🏦 Loan Approval Predictor")

# Form inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Income", min_value=0, step=1000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
loan_amnt = st.number_input("Loan Amount", min_value=0, step=1000)
loan_intent = st.selectbox("Loan Intent", ["VENTURE", "EDUCATION", "MEDICAL", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.number_input("Interest Rate", min_value=0.0, step=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0)
credit_score = st.number_input("Credit Score", min_value=0)
previous_loan_defaults_on_file = st.selectbox("Previous loan?", ["Yes", "No"])

if st.button("Predict"):
    input_data = [[
        person_age, person_gender, person_education, person_income,
        person_home_ownership, loan_amnt, loan_intent,
        loan_int_rate, cb_person_cred_hist_length,
        credit_score, previous_loan_defaults_on_file
    ]]

    input_df = pd.DataFrame(input_data, columns=[
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate','cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ])

    st.subheader("🔍 Input Preview")
    st.dataframe(input_df)

    prediction = model.predict(input_df)[0]
    st.write("Prediction Value:", prediction)
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")
