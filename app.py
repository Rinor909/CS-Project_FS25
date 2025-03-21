import streamlit as st
import numpy as np

st.title("✈️ Flight Ticket Price Predictor")

# Airline selection
airline = st.selectbox("Select Airline", df["Airline"].unique())
distance = st.number_input("Flight Distance (Miles)", min_value=100, max_value=5000)
departure_time = st.selectbox("Departure Time", df["Departure Time"].unique())
duration_hours = st.number_input("Flight Duration (Hours)", min_value=1, max_value=15)
duration_minutes = duration_hours * 60
flight_class = st.selectbox("Select Class", df["Class"].unique())

# Predict button
if st.button("Predict Ticket Price"):
    input_data = np.array([[airline, distance, departure_time, duration_minutes, flight_class]])
    predicted_price = model.predict(input_data)
    st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")