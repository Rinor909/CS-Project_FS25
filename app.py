import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static
import json
import random

# Page configuration
st.set_page_config(page_title="Flight Price Predictor", layout="wide")

# Function to suppress print statements
import sys
import io

# Redirect stdout to suppress print statements
class DummyFile(io.StringIO):
    def write(self, x):
        pass

original_stdout = sys.stdout
sys.stdout = DummyFile()

# Google Drive file ID (Extracted from shared link)
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

# Create sample city coordinates data if not exists
@st.cache_data
def get_city_coordinates():
    if not os.path.exists(city_coords_file):
        # Sample coordinates for major US cities
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
            "Detroit": [42.3314, -83.0458],
            "Minneapolis": [44.9778, -93.2650],
            "Portland": [45.5051, -122.6750],
            "Las Vegas": [36.1699, -115.1398],
            "Nashville": [36.1627, -86.7816],
            "Baltimore": [39.2904, -76.6122],
            "Washington": [38.9072, -77.0369],
            "St. Louis": [38.6270, -90.1994],
            "Orlando": [28.5383, -81.3792],
            "Charlotte": [35.2271, -80.8431],
            # Add specific cities that appear in the interface
            "Nantucket": [41.2835, -70.0995],
            "Tampa": [27.9506, -82.4572],
        }
        
        with open(city_coords_file, 'w') as f:
            json.dump(city_coords, f)
        return city_coords
    else:
        with open(city_coords_file, 'r') as f:
            city_data = json.load(f)
            
            # Ensure critical cities are in the coordinates list
            if "Nantucket" not in city_data:
                city_data["Nantucket"] = [41.2835, -70.0995]
            if "Tampa" not in city_data:
                city_data["Tampa"] = [27.9506, -82.4572]
                
            # Save updated coordinates
            with open(city_coords_file, 'w') as f:
                json.dump(city_data, f)
                
            return city_data

# Load city coordinates
city_coords = get_city_coordinates()

# Generate coordinates for cities not in our database
def get_coordinates_for_city(city_name):
    # Log original city name for debugging
    original_city = city_name
    
    # Extract the base city name (before commas or parentheses)
    base_city = city_name.split(',')[0].split('(')[0].strip()
    
    # Check for direct match
    if city_name in city_coords:
        return city_coords[city_name]
    
    # Check for match with just the base city name
    if base_city in city_coords:
        return city_coords[base_city]
    
    # Check for specific known cities
    if "Nantucket" in city_name:
        return city_coords["Nantucket"]
    elif "Tampa" in city_name:
        return city_coords["Tampa"]
    
    # Check for partial matches
    for known_city, coords in city_coords.items():
        if base_city in known_city or known_city in base_city:
            return coords
    
    # If no match, generate random coordinates within continental US
    lat = random.uniform(25, 49)  # Continental US latitude range
    lng = random.uniform(-125, -65)  # Continental US longitude range
    
    # Save for future use
    city_coords[city_name] = [lat, lng]
    with open(city_coords_file, 'w') as f:
        json.dump(city_coords, f)
    
    # Debug info
    print(f"Generated random coordinates for {original_city}: [{lat}, {lng}]")
    
    return [lat, lng]

# Create a flight route map
def create_flight_map(departure_city, arrival_city):
    # Get coordinates
    dep_coords = get_coordinates_for_city(departure_city)
    arr_coords = get_coordinates_for_city(arrival_city)
    
    # Debug log for coordinates
    print(f"Departure: {departure_city} => {dep_coords}")
    print(f"Arrival: {arrival_city} => {arr_coords}")
    
    # Create map centered between departure and arrival
    center_lat = (dep_coords[0] + arr_coords[0]) / 2
    center_lng = (dep_coords[1] + arr_coords[1]) / 2
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4)
    
    # Add markers for departure and arrival
    folium.Marker(
        dep_coords, 
        popup=departure_city, 
        icon=folium.Icon(color="green", icon="plane-departure", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        arr_coords, 
        popup=arrival_city, 
        icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa")
    ).add_to(m)
    
    # Add a line connecting the two cities
    folium.PolyLine(
        locations=[dep_coords, arr_coords],
        color="blue",
        weight=3,
        opacity=0.7,
        dash_array="5",
    ).add_to(m)
    
    # Add a curved line to represent the flight path
    # Calculate an intermediate point that's above the straight line for the curve
    mid_lat = (dep_coords[0] + arr_coords[0]) / 2
    mid_lng = (dep_coords[1] + arr_coords[1]) / 2
    
    # Push the midpoint slightly north to create a curved appearance
    mid_lat_offset = mid_lat + (abs(dep_coords[1] - arr_coords[1]) / 10)
    
    # Create the curved line
    folium.PolyLine(
        locations=[dep_coords, [mid_lat_offset, mid_lng], arr_coords],
        color="red",
        weight=4,
        opacity=0.8,
    ).add_to(m)
    
    return m

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(output)
        
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
            return None, None, None, None, None
            
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
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# Add some custom CSS
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
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    # Load data (while suppressing print statements)
    df, label_encoders, departure_cities, arrival_cities, airlines = load_data()
    
    # Restore stdout for debugging
    sys.stdout = original_stdout
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path and format.")
        st.stop()
    
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
    
    with col2:
        # Departure city selection
        departure_city = st.selectbox("Departure City", options=departure_cities)
    
    with col3:
        # Arrival city selection
        arrival_city = st.selectbox("Arrival City", options=arrival_cities, 
                                  index=min(1, len(arrival_cities)-1))  # Default to second city if available
    
    # Predict button in its own row
    if st.button("Predict Ticket Price & Show Route", type="primary"):
        try:
            # Create columns for results display
            map_col, info_col = st.columns([3, 2])
            
            with map_col:
                st.subheader("Flight Route")
                # Enable print statements for debugging
                sys.stdout = original_stdout
                
                # Create and display map
                flight_map = create_flight_map(departure_city, arrival_city)
                folium_static(flight_map, width=600, height=400)
            
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
                    # Otherwise use model to predict
                    # Convert selections to encoded values
                    airline_encoded = label_encoders["Airline"].transform([airline])[0]
                    departure_city_encoded = label_encoders["Departure City"].transform([departure_city])[0]
                    arrival_city_encoded = label_encoders["Arrival City"].transform([arrival_city])[0]
                    
                    # Calculate approximate distance based on similar routes
                    similar_departures = df[df["Departure City"] == departure_city]
                    similar_arrivals = df[df["Arrival City"] == arrival_city]
                    
                    if not similar_departures.empty and not similar_arrivals.empty:
                        avg_distance = (similar_departures["Distance"].mean() + similar_arrivals["Distance"].mean()) / 2
                    else:
                        avg_distance = df["Distance"].mean()
                    
                    # Calculate direct distance based on coordinates
                    dep_coords = get_coordinates_for_city(departure_city)
                    arr_coords = get_coordinates_for_city(arrival_city)
                    
                    # Helper function to calculate distance between coordinates
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        from math import radians, cos, sin, asin, sqrt
                        # Convert latitude and longitude from degrees to radians
                        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        # Radius of Earth in miles
                        r = 3956
                        return c * r
                    
                    # Calculate direct distance
                    direct_distance = haversine_distance(
                        dep_coords[0], dep_coords[1], 
                        arr_coords[0], arr_coords[1]
                    )
                    
                    # Use the calculated distance if available
                    if direct_distance > 0:
                        avg_distance = direct_distance
                    
                    # Create and train model
                    X = df[["Airline_encoded", "Distance", "Departure City_encoded", "Arrival City_encoded"]]
                    y = df["Fare"]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Make prediction
                    input_data = np.array([[airline_encoded, avg_distance, departure_city_encoded, arrival_city_encoded]])
                    predicted_price = model.predict(input_data)
                    
                    # Display prediction
                    st.markdown(f"<div class='price-display'>${predicted_price[0]:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<p>Predicted price (new route)</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Estimated distance: {avg_distance:.0f} miles</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show flight details
                st.subheader("Flight Details")
                st.markdown(f"""
                * **Airline**: {airline}
                * **Route**: {departure_city} to {arrival_city}
                * **Distance**: {distance if 'distance' in locals() else avg_distance:.0f} miles
                * **Price**: {'$' + str(round(avg_fare, 2)) if 'avg_fare' in locals() else '$' + str(round(predicted_price[0], 2))}
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
            # Print the full exception traceback for debugging
            import traceback
            st.write(traceback.format_exc())

if __name__ == "__main__":
    main()