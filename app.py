import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
import sys
import io
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pydeck as pdk

# Set Streamlit page configuration
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Suppress unwanted print outputs
class DummyFile(io.StringIO):
    def write(self, x):
        pass
original_stdout = sys.stdout
sys.stdout = DummyFile()

# Google Drive file ID and local filename
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"
output = "US_Airline_Flight_Routes_and_Fares.csv"
city_coords_file = "us_city_coordinates.json"

# Download dataset from Google Drive using requests if not already present
if not os.path.exists(output):
    st.warning("⚠️ Downloading dataset from Google Drive...")
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    with open(output, "wb") as file:
        file.write(response.content)
    st.success("✅ File downloaded successfully!")

# ---------- City Coordinates Functions (for 3D Map) ----------

@st.cache_data
def get_city_coordinates():
    # If coordinates file doesn't exist, create a minimal dictionary
    if not os.path.exists(city_coords_file):
        city_coords = {
            "New York": [40.7128, -74.0060],
            "Los Angeles": [34.0522, -118.2437],
            "Chicago": [41.8781, -87.6298],
            "Houston": [29.7604, -95.3698],
            "Phoenix": [33.4484, -112.0740],
            "San Francisco": [37.7749, -122.4194],
            "Miami": [25.7617, -80.1918]
        }
        with open(city_coords_file, 'w') as f:
            json.dump(city_coords, f)
        return city_coords
    else:
        with open(city_coords_file, 'r') as f:
            return json.load(f)

city_coords = get_city_coordinates()

def get_coordinates_for_city(city_name, city_coords=city_coords):
    # First, try direct match
    if city_name in city_coords:
        return city_coords[city_name]
    # Otherwise, try to clean up the city name (e.g., remove extra text)
    base_city = city_name.split(',')[0].split('(')[0].strip()
    if base_city in city_coords:
        return city_coords[base_city]
    return None

def city_has_coordinates(city_name):
    return get_coordinates_for_city(city_name) is not None

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    c = 2 * asin(sqrt(a)) 
    r = 3956  # Radius of earth in miles
    return c * r

def create_arc_path(start_coords, end_coords, steps=20):
    points = []
    for i in range(steps + 1):
        t = i / steps
        lat = start_coords[0] + t * (end_coords[0] - start_coords[0])
        lng = start_coords[1] + t * (end_coords[1] - start_coords[1])
        altitude = 0
        if 0 < i < steps:
            normalized_t = (t - 0.5) * 2
            altitude = 100000 * (1 - normalized_t**2)
        points.append({"position": [lng, lat, altitude], "t": t})
    return points

def create_3d_flight_map(departure_city, arrival_city):
    dep_coords = get_coordinates_for_city(departure_city)
    arr_coords = get_coordinates_for_city(arrival_city)
    if dep_coords is None or arr_coords is None:
        st.error("Coordinates not found for one of the selected cities.")
        return None
    
    # Arc Layer: flight route
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=[{
            "source": dep_coords,
            "target": arr_coords,
            "source_name": departure_city,
            "target_name": arrival_city
        }],
        get_source_position=["source[1]", "source[0]"],
        get_target_position=["target[1]", "target[0]"],
        get_width=5,
        get_color=[0, 128, 200, 140],
        pickable=True,
    )
    
    # Scatter Layer: Departure and Arrival Cities
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {"position": [dep_coords[1], dep_coords[0]], "name": departure_city},
            {"position": [arr_coords[1], arr_coords[0]], "name": arrival_city}
        ],
        get_position="position",
        get_radius=10000,
        get_fill_color=[255, 0, 0, 200],
        pickable=True,
    )
    
    # Calculate center of the route for view state
    center_lat = (dep_coords[0] + arr_coords[0]) / 2
    center_lng = (dep_coords[1] + arr_coords[1]) / 2
    distance = haversine_distance(dep_coords[0], dep_coords[1], arr_coords[0], arr_coords[1])
    zoom_level = 4
    if distance < 200:
        zoom_level = 7
    elif distance < 500:
        zoom_level = 6
    elif distance < 1000:
        zoom_level = 5
    elif distance > 2000:
        zoom_level = 3
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=zoom_level,
        pitch=45,
        bearing=0
    )
    tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
    
    deck = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    return deck

# Restore stdout for debugging
sys.stdout = original_stdout

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(output, dtype=str)  # Load everything as string initially
        st.write("📌 Columns in dataset:", df.columns.tolist())
        
        # Map columns from dataset to expected names
        column_mapping = {
            "carrier_lg": "Airline",
            "nsmiles": "Distance",
            "fare": "Fare",
            "city1": "Departure City",
            "city2": "Arrival City"
        }
        df = df.rename(columns=column_mapping)
        
        # Select relevant columns
        selected_columns = ["Airline", "Distance", "Fare", "Departure City", "Arrival City"]
        df = df[selected_columns]
        
        # Convert numeric columns to proper types
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        df.dropna(inplace=True)
        
        # Preserve full airline names for UI while encoding internally
        # Create an encoder for Airline and add a new column "Airline_encoded"
        le_airline = LabelEncoder()
        df["Airline_encoded"] = le_airline.fit_transform(df["Airline"])
        
        # Similarly encode Departure and Arrival Cities (but keep original names for display)
        le_dep = LabelEncoder()
        df["Departure City_encoded"] = le_dep.fit_transform(df["Departure City"])
        le_arr = LabelEncoder()
        df["Arrival City_encoded"] = le_arr.fit_transform(df["Arrival City"])
        
        # Store unique lists for dropdowns (full names)
        departure_cities = df["Departure City"].unique().tolist()
        arrival_cities = df["Arrival City"].unique().tolist()
        airlines = df["Airline"].unique().tolist()
        
        # Prepare a dictionary of encoders to convert UI selections for model prediction
        label_encoders = {
            "Airline": le_airline,
            "Departure City": le_dep,
            "Arrival City": le_arr
        }
        
        return df, label_encoders, departure_cities, arrival_cities, airlines
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

df, label_encoders, departure_cities, arrival_cities, airlines = load_data()

def main():
    st.title("✈️ Flight Ticket Price Predictor")
    st.markdown("""
    This app predicts flight ticket prices based on historical data and displays the flight route on a 3D map.
    """)
    
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Airline selection: show full names
        selected_airline = st.selectbox("Select Airline", options=airlines)
        airline_encoded = label_encoders["Airline"].transform([selected_airline])[0]
    
    with col2:
        # Departure city selection: show full city names
        selected_dep_city = st.selectbox("Departure City", options=departure_cities)
    
    with col3:
        # Arrival city selection: show full city names
        selected_arr_city = st.selectbox("Arrival City", options=arrival_cities)
    
    distance_input = st.number_input("Enter Flight Distance (Miles)", min_value=100, max_value=5000, value=1000)
    
    # Predict button and map display
    if st.button("Predict Ticket Price & Show Route"):
        try:
            # Convert selected cities to encoded values
            dep_city_encoded = label_encoders["Departure City"].transform([selected_dep_city])[0]
            arr_city_encoded = label_encoders["Arrival City"].transform([selected_arr_city])[0]
            
            # Train a model using the encoded columns
            X = df[["Airline_encoded", "Distance", "Departure City_encoded", "Arrival City_encoded"]]
            y = df["Fare"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Prepare input for prediction
            input_data = np.array([[airline_encoded, distance_input, dep_city_encoded, arr_city_encoded]])
            predicted_price = model.predict(input_data)
            
            st.markdown(f"<div style='font-size: 28px; font-weight: bold; color: #1E88E5;'>Predicted Ticket Price: ${predicted_price[0]:.2f}</div>", unsafe_allow_html=True)
            
            # Display 3D flight route map
            st.subheader("Flight Route (3D Map)")
            flight_map = create_3d_flight_map(selected_dep_city, selected_arr_city)
            if flight_map:
                st.pydeck_chart(flight_map)
            
            # Display additional flight details
            st.subheader("Flight Details")
            st.markdown(f"""
            * **Airline:** {selected_airline}
            * **Route:** {selected_dep_city} → {selected_arr_city}
            * **Distance:** {distance_input} miles
            * **Predicted Fare:** ${predicted_price[0]:.2f}
            """)
            
            # Show similar routes for comparison
            st.subheader("Similar Routes")
            similar_routes = df[
                (df["Airline"] == selected_airline) & 
                ((df["Departure City"] == selected_dep_city) | (df["Arrival City"] == selected_arr_city))
            ][["Departure City", "Arrival City", "Distance", "Fare"]].head(5)
            if not similar_routes.empty:
                st.dataframe(similar_routes)
            else:
                similar_routes = df[df["Airline"] == selected_airline][["Departure City", "Arrival City", "Distance", "Fare"]].head(5)
                st.write("No direct matches found. Here are some routes by the same airline:")
                st.dataframe(similar_routes)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.write(traceback.format_exc())

if __name__ == "__main__":
    main()