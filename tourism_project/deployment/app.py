import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "tourism_model.joblib")
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to predict whether the customer is likely "
    "to purchase the tourism package."
)

age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox(
    "Type of Contact", ["Self Enquiry", "Company Invited"]
)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input(
    "Duration of Pitch", min_value=0.0, value=10.0
)
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
)
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input(
    "Number of Persons Visiting", min_value=1, value=3
)
number_of_followups = st.number_input(
    "Number of Followups", min_value=0.0, value=3.0
)
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
)
preferred_property_star = st.selectbox(
    "Preferred Property Star", [3.0, 4.0, 5.0]
)
marital_status = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced"]
)
number_of_trips = st.number_input(
    "Number of Trips", min_value=0.0, value=3.0
)
passport = st.selectbox(
    "Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)
own_car = st.selectbox(
    "Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
number_of_children_visiting = st.number_input(
    "Number of Children Visiting", min_value=0.0, value=0.0
)
designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
monthly_income = st.number_input(
    "Monthly Income", min_value=0.0, value=20000.0
)

# Retained during model training; kept at a neutral value for prediction.
unnamed_0 = 0

if st.button("Predict Purchase"):
    input_data = pd.DataFrame([{
        "Unnamed: 0": unnamed_0,
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            "The model predicts that the customer is likely to purchase the package."
        )
    else:
        st.info(
            "The model predicts that the customer is unlikely to purchase the package."
        )

    st.write(f"Estimated probability of purchase: **{probability:.2%}**")
