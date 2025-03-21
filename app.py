import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data  # Cache to prevent reloading on every interaction
def load_data():
    df = pd.read_csv('/kaggle/input/us-airline-flight-routes-and-fares-1993-2024/US Airline Flight Routes and Fares 1993-2024.csv', low_memory=False)

    # Preprocess data
    df["Airline"] = df["Airline"].astype("category").cat.codes
    df["Departure Time"] = df["Departure Time"].astype("category").cat.codes
    df["Class"] = df["Class"].astype("category").cat.codes

    # Convert 'Duration' to total minutes
    def convert_duration(duration):
        try:
            h, m = map(int, duration.replace("h", "").replace("m", "").split())
            return h * 60 + m  # Convert to total minutes
        except:
            return None  # Handle invalid values

    df["Duration"] = df["Duration"].apply(convert_duration)
    df.dropna(inplace=True)  # Remove invalid rows

    return df

df = load_data()  # Load the dataset once

st.title("✈️ Flight Ticket Price Predictor")

# Airline selection
airline = st.selectbox("Select Airline", df["Airline"].unique())
distance = st.number_input("Flight Distance (Miles)", min_value=100, max_value=5000)
departure_time = st.selectbox("Departure Time", df["Departure Time"].unique())
duration_hours = st.number_input("Flight Duration (Hours)", min_value=1, max_value=15)

# Predict button
if st.button("Predict Ticket Price"):
    input_data = np.array([[airline, distance, departure_time, duration_minutes, flight_class]])
    predicted_price = model.predict(input_data)
    st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")