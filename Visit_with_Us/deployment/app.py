import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Predictor",
                   page_icon="🌿", layout="wide")

MODEL_REPO = "anuragmishrarock/tourism-wellness-model"

@st.cache_resource
def load_model():
    model_path   = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
    feature_path = hf_hub_download(repo_id=MODEL_REPO, filename="feature_names.json")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feature_path, 'r') as f:
        features = json.load(f)
    return model, features

model, feature_names = load_model()

st.title("🌿 Wellness Tourism Package Predictor")
st.markdown("**Visit with Us** — Predict whether a customer will purchase the Wellness Tourism Package")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Details")
    age             = st.slider("Age", 18, 70, 35)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation      = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income  = st.number_input("Monthly Income (₹)", 10000, 100000, 25000, step=1000)

with col2:
    st.subheader("✈️ Trip Preferences")
    city_tier                  = st.selectbox("City Tier", [1, 2, 3])
    number_of_trips            = st.slider("Number of Trips / Year", 1, 10, 2)
    number_of_person_visiting  = st.slider("Persons Visiting", 1, 5, 2)
    number_of_children         = st.slider("Children (< 5 yrs)", 0, 3, 0)
    preferred_property_star    = st.selectbox("Preferred Hotel Stars", [3, 4, 5])
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car  = st.selectbox("Owns Car?",     [0, 1], format_func=lambda x: "Yes" if x else "No")

with col3:
    st.subheader("📞 Interaction Data")
    type_of_contact        = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched        = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch      = st.slider("Duration of Pitch (min)", 5, 60, 15)
    number_of_followups    = st.slider("Number of Follow-ups", 1, 6, 3)
    pitch_satisfaction     = st.slider("Pitch Satisfaction Score", 1, 5, 3)

st.divider()

if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True):
    input_data = {
        'Age':                       age,
        'TypeofContact':             0 if type_of_contact == "Company Invited" else 1,
        'CityTier':                  city_tier,
        'DurationOfPitch':           duration_of_pitch,
        'Gender':                    0 if gender == "Female" else 1,
        'NumberOfPersonVisiting':    number_of_person_visiting,
        'NumberOfFollowups':         number_of_followups,
        'PreferredPropertyStar':     preferred_property_star,
        'NumberOfTrips':             number_of_trips,
        'Passport':                  passport,
        'PitchSatisfactionScore':    pitch_satisfaction,
        'OwnCar':                    own_car,
        'NumberOfChildrenVisiting':  number_of_children,
        'MonthlyIncome':             monthly_income,
    }
    for occ in ["Free Lancer", "Large Business", "Salaried", "Small Business"]:
        input_data[f"Occupation_{occ}"] = 1 if occupation == occ else 0
    for prod in ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]:
        input_data[f"ProductPitched_{prod}"] = 1 if product_pitched == prod else 0
    for ms in ["Divorced", "Married", "Single", "Unmarried"]:
        input_data[f"MaritalStatus_{ms}"] = 1 if marital_status == ms else 0
    for des in ["AVP", "Executive", "Manager", "Senior Manager", "VP"]:
        input_data[f"Designation_{des}"] = 1 if designation == des else 0

    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if prediction == 1:
            st.success("✅ **LIKELY TO PURCHASE**")
        else:
            st.error("❌ **UNLIKELY TO PURCHASE**")
        st.metric("Purchase Probability", f"{probability:.1%}")
        st.progress(float(probability))

    if probability > 0.7:
        st.info("💡 **High Priority** — Recommend immediate follow-up with a personalised offer.")
    elif probability > 0.4:
        st.info("💡 **Medium Priority** — Schedule a follow-up highlighting package benefits.")
    else:
        st.info("💡 **Low Priority** — Include in future broader campaigns.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("Algorithm: Gradient Boosting")
st.sidebar.markdown("ROC-AUC: 0.9534")
st.sidebar.markdown("Built by: anuragmishrarock")
