import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Google Drive file ID (Extracted from shared link)
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"
output = "US_Airline_Flight_Routes_and_Fares.csv"

# Download dataset from Google Drive (Using requests instead of gdown)
if not os.path.exists(output):
    st.warning("⚠️ Downloading dataset from Google Drive...")
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    with open(output, "wb") as file:
        file.write(response.content)
    st.success("✅ File downloaded successfully!")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(output, dtype=str)  # Load everything as string first

    # Print available columns for debugging
    st.write("📌 Columns in dataset:", df.columns.tolist())

    # Rename columns to match expected format
    column_mapping = {
        "carrier_lg": "Airline",
        "nsmiles": "Distance",
        "fare": "Fare",
        "city1": "Departure City",
        "city2": "Arrival City"
    }
    df = df.rename(columns=column_mapping)

    # Check if renamed columns exist
    expected_columns = ["Airline", "Distance", "Fare"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        st.error(f"🚨 Missing renamed columns: {missing_columns}. Please check the dataset format.")
        return None

    # Convert categorical columns
    categorical_columns = ["Airline"]
    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes

    # Convert numeric columns
    numeric_cols = ["Distance", "Fare"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)  # Remove invalid rows

    return df

df = load_data()  # Load the dataset once

if df is not None:
    # Train a simple Linear Regression model
    X = df[["Airline", "Distance", "Departure Time", "Duration", "Class"]]
    y = df["Fare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Streamlit UI
    st.title("✈️ Flight Ticket Price Predictor")

    # User input fields
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