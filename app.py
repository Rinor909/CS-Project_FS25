import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import json
from math import radians, cos, sin, asin, sqrt

# Page configuration
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Google Drive file ID
file_id = "11NgU1kWQIAzBhEbG3L6XsLRqm1T2dn4I"
output = "US_Airline_Flight_Routes_and_Fares.csv"
city_coords_file = "us_city_coordinates.json"

# Download dataset from Google Drive if it doesn't exist
if not os.path.exists(output):
    with st.spinner("⚠️ Downloading dataset from Google Drive..."):
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url)
        with open(output, "wb") as file:
            file.write(response.content)
        st.success("✅ File downloaded successfully!")

# Improved city coordinates data
@st.cache_data
def get_city_coordinates():
    if not os.path.exists(city_coords_file):
        # Basic coordinates for US cities
        city_coords = {
            "New York": [40.7128, -74.0060],
            "Los Angeles": [34.0522, -118.2437],
            "Chicago": [41.8781, -87.6298],
            "Houston": [29.7604, -95.3698],
            "Phoenix": [33.4484, -112.0740],
            "Philadelphia": [39.9526, -75.1652],
            "San Antonio": [29.4241, -98.4936],
            "San Diego": [32.7157, -117.1611],
            "Dallas": [32.7767, -96.7970],
            "San Francisco": [37.7749, -122.4194],
            "Seattle": [47.6062, -122.3321],
            "Denver": [39.7392, -104.9903],
            "Boston": [42.3601, -71.0589],
            "Atlanta": [33.7490, -84.3880],
            "Miami": [25.7617, -80.1918],
            "Dallas/Fort Worth": [32.8998, -97.0403],
            "Washington": [38.9072, -77.0369],
            "Las Vegas": [36.1699, -115.1398],
            "Nashville": [36.1627, -86.7816],
            "Tampa": [27.9506, -82.4572],
        }
        
        with open(city_coords_file, 'w') as f:
            json.dump(city_coords, f)
        return city_coords
    else:
        with open(city_coords_file, 'r') as f:
            return json.load(f)

# Load city coordinates
city_coords = get_city_coordinates()

# Get coordinates for city
def get_coordinates_for_city(city_name, city_coords=city_coords):
    # Handle special case for "Dallas/Fort Worth"
    if "Dallas/Fort Worth" in city_name:
        return city_coords.get("Dallas/Fort Worth", [32.8998, -97.0403])
    
    # Remove state abbreviations and other extras
    base_city = city_name.split(',')[0].split('(')[0].strip()
    
    # Direct match with full name
    if city_name in city_coords:
        return city_coords[city_name]
    
    # Match with base name
    if base_city in city_coords:
        return city_coords[base_city]
    
    # If no match found, return None
    return None

# Function to check if a city exists in coordinates
def city_has_coordinates(city_name):
    return get_coordinates_for_city(city_name) is not None

# Haversine distance calculator
def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of Earth in miles
    return c * r

# Function to create points along an arc path
def create_arc_path(start_coords, end_coords, steps=20):
    points = []
    for i in range(steps + 1):
        t = i / steps
        lat = start_coords[0] + t * (end_coords[0] - start_coords[0])
        lng = start_coords[1] + t * (end_coords[1] - start_coords[1])
        
        altitude = 0
        if i > 0 and i < steps:
            normalized_t = (t - 0.5) * 2  # -1 to 1
            altitude = 100000 * (1 - normalized_t**2)  # Higher in the middle
        
        points.append({"position": [lng, lat, altitude], "t": t})
    
    return points

# 3D flight route visualization with PyDeck
def create_3d_flight_map(departure_city, arrival_city):
    # Get coordinates
    dep_coords = get_coordinates_for_city(departure_city)
    arr_coords = get_coordinates_for_city(arrival_city)
    
    if not dep_coords or not arr_coords:
        st.error(f"Cannot display map: Missing coordinates for {departure_city if not dep_coords else arrival_city}")
        return None
    
    # Create arc layer for the flight path
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
        get_height=0.5,
        get_tilt=15,
        get_source_color=[0, 255, 0, 200],  # Green for departure
        get_target_color=[255, 0, 0, 200],  # Red for arrival
        pickable=True,
    )
    
    # Create scatter plot for departure and arrival cities
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {"position": [dep_coords[1], dep_coords[0]], "name": departure_city, "size": 10000, "color": [0, 255, 0]},
            {"position": [arr_coords[1], arr_coords[0]], "name": arrival_city, "size": 10000, "color": [255, 0, 0]}
        ],
        get_position="position",
        get_radius="size",
        get_fill_color="color",
        pickable=True,
    )
    
    # Calculate center and zoom level
    center_lat = (dep_coords[0] + arr_coords[0]) / 2
    center_lng = (dep_coords[1] + arr_coords[1]) / 2
    
    # Adjust zoom based on distance
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
    
    # Create the 3D view
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=zoom_level,
        pitch=45,  # Tilted view for 3D effect
        bearing=0
    )
    
    # Create tooltip for interactive elements
    tooltip = {
        "html": "<b>{name}</b>",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    # Combine all layers
    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    
    return r

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(output)
        
        # Column mapping
        column_mapping = {
            "carrier_lg": "Airline",
            "nsmiles": "Distance",
            "fare": "Fare",
            "city1": "Departure City",
            "city2": "Arrival City"
        }
        
        # Check if expected columns exist
        missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {missing_columns}")
            return None, None, None, None, None
            
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select relevant columns
        selected_columns = ["Airline", "Distance", "Fare", "Departure City", "Arrival City"]
        df = df[selected_columns]
        
        # Convert numeric columns
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        
        # Handle missing values
        df.dropna(inplace=True)
        
        # Find the 5 biggest airlines based on frequency
        top_airlines = df["Airline"].value_counts().head(5).index.tolist()
        
        # Filter to only include the top 5 airlines
        df = df[df["Airline"].isin(top_airlines)]
        
        # Filter cities to only those with known coordinates
        valid_cities = []
        for city in set(df["Departure City"].unique()).union(set(df["Arrival City"].unique())):
            if city_has_coordinates(city):
                valid_cities.append(city)
        
        # Filter to only include routes where both cities have coordinates
        df = df[df["Departure City"].isin(valid_cities) & df["Arrival City"].isin(valid_cities)]
        
        # Create encoders dictionary
        label_encoders = {}
        
        # Store unique values for city selections before encoding
        departure_cities = df["Departure City"].unique().tolist()
        arrival_cities = df["Arrival City"].unique().tolist()
        airlines = top_airlines
        
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

# Add custom CSS
st.markdown("""
<style>
    .route-info {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .price-display {
        font-size: 28px;
        font-weight: bold;
        color: #1E88E5;
    }
    .airline-logo {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        background-color: #1E88E5;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    # Load data
    df, label_encoders, departure_cities, arrival_cities, airlines = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path and format.")
        return
    
    # Display app title and description
    st.title("✈️ Flight Ticket Price Predictor")
    st.markdown("""
    This app predicts flight ticket prices based on airline and route information.
    Select your preferences below to get a price estimate and view the route on a map.
    """)
    
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Airline selection
        airline = st.selectbox("Select Airline", options=airlines)
    
    # Get valid departure cities for this airline
    valid_departure_cities = df[df["Airline"] == airline]["Departure City"].unique().tolist()
    
    with col2:
        # Departure city selection
        departure_city = st.selectbox("Departure City", options=valid_departure_cities)
    
    # Get valid arrival cities for this airline and departure city
    valid_arrival_cities = df[(df["Airline"] == airline) & 
                             (df["Departure City"] == departure_city)]["Arrival City"].unique().tolist()
    
    # If no valid arrival cities, use all arrival cities for this airline
    if not valid_arrival_cities:
        valid_arrival_cities = df[df["Airline"] == airline]["Arrival City"].unique().tolist()
    
    with col3:
        # Arrival city selection
        arrival_city = st.selectbox(
            "Arrival City", 
            options=valid_arrival_cities,
            index=min(0, len(valid_arrival_cities)-1) if valid_arrival_cities else 0
        )
    
    # Predict button
    if st.button("Predict Ticket Price & Show Route", type="primary"):
        try:
            # Create columns for results display
            map_col, info_col = st.columns([3, 2])
            
            with map_col:
                st.subheader("Flight Route (3D View)")
                
                # Create and display 3D map with PyDeck
                flight_map = create_3d_flight_map(departure_city, arrival_city)
                if flight_map:
                    st.pydeck_chart(flight_map)
                    st.caption("Tip: Click and drag to rotate the view. Scroll to zoom in/out.")
            
            with info_col:
                # Filter data for the selected route
                route_data = df[
                    (df["Airline"] == airline) & 
                    (df["Departure City"] == departure_city) & 
                    (df["Arrival City"] == arrival_city)
                ]
                
                st.markdown("<div class='route-info'>", unsafe_allow_html=True)
                st.markdown(f"<div class='airline-logo'>{airline[0]}</div> <b>{airline}</b>", unsafe_allow_html=True)
                st.markdown(f"<h3>{departure_city} → {arrival_city}</h3>", unsafe_allow_html=True)
                
                if len(route_data) > 0:
                    # If we have exact route data, use the average fare
                    avg_fare = route_data["Fare"].mean()
                    distance = route_data["Distance"].mean()
                    
                    st.markdown(f"<div class='price-display'>${avg_fare:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p>Based on {len(route_data)} existing flights</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Distance: {distance:.0f} miles</p>", unsafe_allow_html=True)
                else:
                    # Use model to predict
                    # Convert selections to encoded values
                    airline_encoded = label_encoders["Airline"].transform([airline])[0]
                    departure_city_encoded = label_encoders["Departure City"].transform([departure_city])[0]
                    arrival_city_encoded = label_encoders["Arrival City"].transform([arrival_city])[0]
                    
                    # Calculate direct distance
                    dep_coords = get_coordinates_for_city(departure_city)
                    arr_coords = get_coordinates_for_city(arrival_city)
                    
                    direct_distance = 0
                    if dep_coords and arr_coords:
                        direct_distance = haversine_distance(
                            dep_coords[0], dep_coords[1], 
                            arr_coords[0], arr_coords[1]
                        )
                    
                    # Use the calculated distance
                    avg_distance = direct_distance if direct_distance > 0 else df["Distance"].mean()
                    
                    # Create and train model
                    X = df[["Airline_encoded", "Distance", "Departure City_encoded", "Arrival City_encoded"]]
                    y = df["Fare"]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Make prediction
                    input_data = np.array([[airline_encoded, avg_distance, departure_city_encoded, arrival_city_encoded]])
                    predicted_price = model.predict(input_data)
                    
                    # Ensure the predicted price is positive
                    predicted_price = max(predicted_price[0], 50.0)  # Minimum fare of $50
                    
                    # Display prediction
                    st.markdown(f"<div class='price-display'>${predicted_price:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<p>Predicted price (new route)</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Estimated distance: {avg_distance:.0f} miles</p>", unsafe_allow_html=True)
                    
                    # Store for later use
                    distance = avg_distance
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show flight details
                st.subheader("Flight Details")
                if len(route_data) > 0:
                    st.markdown(f"""
                    * **Airline**: {airline}
                    * **Route**: {departure_city} to {arrival_city}
                    * **Distance**: {distance:.0f} miles
                    * **Price**: ${avg_fare:.2f}
                    """)
                else:
                    st.markdown(f"""
                    * **Airline**: {airline}
                    * **Route**: {departure_city} to {arrival_city}
                    * **Distance**: {distance:.0f} miles
                    * **Price**: ${predicted_price:.2f} (predicted)
                    """)
            
                # Find similar routes for comparison
                st.subheader("Similar Routes")
                similar_routes = df[
                    (df["Airline"] == airline) & 
                    ((df["Departure City"] == departure_city) | (df["Arrival City"] == arrival_city))
                    ][["Departure City", "Arrival City", "Distance", "Fare"]].head(5)
            
                if not similar_routes.empty:
                    st.dataframe(similar_routes)
                else:
                    similar_routes = df[df["Airline"] == airline][["Departure City", "Arrival City", "Distance", "Fare"]].head(5)
                    st.write("No direct matches found. Here are some routes by the same airline:")
                    st.dataframe(similar_routes)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()