import streamlit as st
import numpy as np

st.title("✈️ Flight Ticket Price Predictor")

# Airline selection
airline = st.selectbox("Select Airline", ["Delta", "United", "American Airlines", "Southwest", "Alaska"])
airline_mapping = {"Delta": 0, "United": 1, "American Airlines": 2, "Southwest": 3, "Alaska": 4}
airline_encoded = airline_mapping[airline]

# Flight Distance
distance = st.number_input("Flight Distance (Miles)", min_value=100, max_value=5000)

# Departure Time
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night"])
departure_time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
departure_time_encoded = departure_time_mapping[departure_time]

# Duration in minutes
duration_hours = st.number_input("Flight Duration (Hours)", min_value=1, max_value=15)
duration_minutes = duration_hours * 60

# Class selection
flight_class = st.selectbox("Select Class", ["Economy", "Business"])
class_mapping = {"Economy": 0, "Business": 1}
class_encoded = class_mapping[flight_class]

# Predict button
if st.button("Predict Ticket Price"):
    input_data = np.array([[airline_encoded, distance, departure_time_encoded, duration_minutes, class_encoded]])
    predicted_price = model.predict(input_data)
    st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")