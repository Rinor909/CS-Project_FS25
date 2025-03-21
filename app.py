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

    # Encode categorical columns for ML but keep original names for dropdowns
    label_encoders = {}
    categorical_columns = ["Airline"]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoder for later use

    # Convert numeric columns correctly
    numeric_cols = ["Distance", "Fare"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)  # Remove invalid rows

    return df, label_encoders

df, label_encoders = load_data()  # Load dataset with label encoders

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
    airline = st.selectbox("Select Airline", label_encoders["Airline"].classes_)  # Show original airline names
    airline_encoded = label_encoders["Airline"].transform([airline])[0]  # Convert to encoded value

    distance = st.number_input("Flight Distance (Miles)", min_value=100, max_value=5000)

    # Keep city names visible in dropdowns
    departure_city_name = st.selectbox("Departure City", df["Departure City"].unique())
    arrival_city_name = st.selectbox("Arrival City", df["Arrival City"].unique())

    # Convert selected cities to their internal values for ML model
    departure_city = df[df["Departure City"] == departure_city_name]["Departure City"].values[0]
    arrival_city = df[df["Arrival City"] == arrival_city_name]["Arrival City"].values[0]

    # Predict button
    if st.button("Predict Ticket Price"):
        input_data = np.array([[airline_encoded, distance, departure_city, arrival_city]])
        predicted_price = model.predict(input_data)
        st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")