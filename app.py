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

    # Correct mapping based on available columns
    column_mapping = {
        "carrier_lg": "Airline",  # Airline column
        "nsmiles": "Distance",  # Distance in miles
        "fare": "Fare",  # Flight fare
        "city1": "Departure City",  # Departure location
        "city2": "Arrival City"  # Arrival location
    }
    df = df.rename(columns=column_mapping)

    # Select only the relevant columns
    selected_columns = ["Airline", "Distance", "Fare", "Departure City", "Arrival City"]
    df = df[selected_columns]

    # Convert categorical columns
    categorical_columns = ["Airline", "Departure City", "Arrival City"]
    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes

    # Convert numeric columns correctly
    numeric_cols = ["Distance", "Fare"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)  # Remove invalid rows

    return df

df = load_data()  # Load the dataset once

if df is not None:
    # Train a simple Linear Regression model
    X = df[["Airline", "Distance", "Departure City", "Arrival City"]]
    y = df["Fare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Streamlit UI
    st.title("✈️ Flight Ticket Price Predictor")

    # User input fields
    airline = st.selectbox("Select Airline", df["Airline"].unique())
    distance = st.number_input("Flight Distance (Miles)", min_value=100, max_value=5000)
    departure_city = st.selectbox("Departure City", df["Departure City"].unique())
    arrival_city = st.selectbox("Arrival City", df["Arrival City"].unique())

    # Predict button
    if st.button("Predict Ticket Price"):
        input_data = np.array([[airline, distance, departure_city, arrival_city]])
        predicted_price = model.predict(input_data)
        st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")