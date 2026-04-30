import streamlit as st
from src.predict_price import predict_house_price

st.set_page_config(
    page_title="House Price Prediction ML",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Real Estate House Price Prediction")
st.write("Predict house prices using Machine Learning.")

area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

location = st.selectbox(
    "Location",
    ["Colombo", "Kandy", "Galle", "Negombo"]
)

house_age = st.number_input("House Age", min_value=0, max_value=100, value=5)
parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1)

if st.button("Predict Price"):
    predicted_price = predict_house_price(
        area,
        bedrooms,
        bathrooms,
        location,
        house_age,
        parking
    )

    st.success(f"Predicted House Price: {predicted_price:,.2f}")