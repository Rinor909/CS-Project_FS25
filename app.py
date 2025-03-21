import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Google Drive file ID (Extracted from shared link)
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"
output = "US_Airline_Flight_Routes_and_Fares.csv"

# Download dataset from Google Drive if it doesn't exist
if not os.path.exists(output):
    with st.spinner("⚠️ Downloading dataset from Google Drive..."):
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url)
        with open(output, "wb") as file:
            file.write(response.content)
        st.success("✅ File downloaded successfully!")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(output)
        
        # Log the columns for debugging
        st.write("📌 Original columns in dataset:", df.columns.tolist())
        
        # Correct mapping based on available columns
        column_mapping = {
            "carrier_lg": "Airline",     # Airline column
            "nsmiles": "Distance",       # Distance in miles
            "fare": "Fare",              # Flight fare
            "city1": "Departure City",   # Departure location
            "city2": "Arrival City"      # Arrival location
        }
        
        # Check if expected columns exist
        missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
        if missing_columns:
            st.error(f"🚨 Missing columns in dataset: {missing_columns}")
            st.stop()
            
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select only the relevant columns
        selected_columns = ["Airline", "Distance", "Fare", "Departure City", "Arrival City"]
        df = df[selected_columns]
        
        # Convert numeric columns
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        
        # Handle missing values
        before_count = len(df)
        df.dropna(inplace=True)
        after_count = len(df)
        if before_count > after_count:
            st.info(f"ℹ️ Removed {before_count - after_count} rows with missing values")
        
        # Create encoders dictionary
        label_encoders = {}
        
        # Store unique values for city selections before encoding
        departure_cities = df["Departure City"].unique().tolist()
        arrival_cities = df["Arrival City"].unique().tolist()
        airlines = df["Airline"].unique().tolist()
        
        # Encode categorical columns
        categorical_columns = ["Airline", "Departure City", "Arrival City"]
        for col in categorical_columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        return df, label_encoders, departure_cities, arrival_cities, airlines
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# Main app function
def main():
    # Load data
    df, label_encoders, departure_cities, arrival_cities, airlines = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path and format.")
        st.stop()
    
    # Display app title and description
    st.title("✈️ Flight Ticket Price Predictor")
    st.markdown("""
    This app predicts flight ticket prices based on airline, distance, and route information.
    Select your preferences below to get a price estimate.
    """)
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Airline selection
        airline = st.selectbox("Select Airline", options=airlines)
        
        # Distance input
        distance = st.number_input(
            "Flight Distance (Miles)", 
            min_value=int(df["Distance"].min()), 
            max_value=int(df["Distance"].max()),
            value=int(df["Distance"].mean())
        )
    
    with col2:
        # City selections
        departure_city = st.selectbox("Departure City", options=departure_cities)
        arrival_city = st.selectbox("Arrival City", options=arrival_cities, 
                                   index=min(1, len(arrival_cities)-1))  # Default to second city if available
    
    # Predict button in its own row
    if st.button("Predict Ticket Price", type="primary"):
        try:
            # Convert selections to encoded values
            airline_encoded = label_encoders["Airline"].transform([airline])[0]
            departure_city_encoded = label_encoders["Departure City"].transform([departure_city])[0]
            arrival_city_encoded = label_encoders["Arrival City"].transform([arrival_city])[0]
            
            # Create and train model
            X = df[["Airline_encoded", "Distance", "Departure City_encoded", "Arrival City_encoded"]]
            y = df["Fare"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make prediction
            input_data = np.array([[airline_encoded, distance, departure_city_encoded, arrival_city_encoded]])
            predicted_price = model.predict(input_data)
            
            # Display prediction
            st.success(f"🛫 Predicted Ticket Price: **${predicted_price[0]:.2f}**")
            
            # Display route summary
            st.info(f"Route: {departure_city} → {arrival_city} ({distance} miles) with {airline}")
            
            # Display model performance
            test_score = model.score(X_test, y_test)
            st.metric("Model Accuracy (R²)", f"{test_score:.2f}")
            
            # Find similar routes for comparison
            st.subheader("Similar Routes")
            similar_routes = df[
                (df["Airline"] == airline) & 
                (df["Distance"].between(distance*0.8, distance*1.2))
            ][["Departure City", "Arrival City", "Distance", "Fare"]].head(5)
            
            if not similar_routes.empty:
                st.dataframe(similar_routes)
            else:
                st.write("No similar routes found in the dataset.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()