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
from flight_visualizations import show_additional_visualizations, create_price_history_chart, add_seasonal_pricing_insight, add_best_booking_time


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

# Improved city coordinates data
@st.cache_data
def get_city_coordinates():
    if not os.path.exists(city_coords_file):
        # Expanded coordinates for US cities (including more cities in various states)
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
            "Nantucket": [41.2835, -70.0995],
            "Tampa": [27.9506, -82.4572],
            "Austin": [30.2672, -97.7431],
            "Columbus": [39.9612, -82.9988],
            "Fort Worth": [32.7555, -97.3308],
            "Dallas/Fort Worth": [32.8998, -97.0403],
            "Indianapolis": [39.7684, -86.1581],
            "Jacksonville": [30.3322, -81.6557],
            "San Jose": [37.3382, -121.8863],
            "Memphis": [35.1495, -90.0490],
            "Louisville": [38.2527, -85.7585],
            "Milwaukee": [43.0389, -87.9065],
            "Kansas City": [39.0997, -94.5786],
            "Albuquerque": [35.0844, -106.6504],
            "Tucson": [32.2226, -110.9747],
            "Fresno": [36.7378, -119.7871],
            "Sacramento": [38.5816, -121.4944],
            "Long Beach": [33.7701, -118.1937],
            "Colorado Springs": [38.8339, -104.8214],
            "Raleigh": [35.7796, -78.6382],
            "Omaha": [41.2565, -95.9345],
            "Oakland": [37.8044, -122.2711],
            "Tulsa": [36.1540, -95.9928],
            "Cleveland": [41.4993, -81.6944],
            "Wichita": [37.6872, -97.3301],
            "Arlington": [32.7357, -97.1081],
            "New Orleans": [29.9511, -90.0715],
            "Honolulu": [21.3069, -157.8583],
            "Anchorage": [61.2181, -149.9003],
            "Salt Lake City": [40.7608, -111.8910],
            "Cincinnati": [39.1031, -84.5120],
            "Pittsburgh": [40.4406, -79.9959],
            "Greensboro": [36.0726, -79.7920],
            "St. Paul": [44.9537, -93.0900],
            "Buffalo": [42.8864, -78.8784],
            "Lexington": [38.0406, -84.5037],
            "Newark": [40.7357, -74.1724],
        }
        
        with open(city_coords_file, 'w') as f:
            json.dump(city_coords, f)
        return city_coords
    else:
        with open(city_coords_file, 'r') as f:
            city_data = json.load(f)
            
            # Add important cities that might be missing
            missing_cities = {
                "Colorado Springs": [38.8339, -104.8214],
                "Dallas/Fort Worth": [32.8998, -97.0403],
                "Fort Worth": [32.7555, -97.3308],
                "Nantucket": [41.2835, -70.0995],
                "Tampa": [27.9506, -82.4572]
            }
            
            for city, coords in missing_cities.items():
                if city not in city_data:
                    city_data[city] = coords
                
            # Save updated coordinates
            with open(city_coords_file, 'w') as f:
                json.dump(city_data, f)
                
            return city_data

# Load city coordinates
city_coords = get_city_coordinates()

# Get coordinates for city
def get_coordinates_for_city(city_name, city_coords=city_coords):
    # Extract the base city name
    if '/' in city_name:
        # Handle special case for "Dallas/Fort Worth"
        if "Dallas/Fort Worth" in city_name:
            return city_coords.get("Dallas/Fort Worth", [32.8998, -97.0403])
        
        parts = city_name.split('/')
        for part in parts:
            clean_part = part.strip().split(',')[0].split('(')[0].strip()
            if clean_part in city_coords:
                return city_coords[clean_part]
    
    # Remove state abbreviations and other extras
    base_city = city_name.split(',')[0].split('(')[0].strip()
    
    # Direct match with full name
    if city_name in city_coords:
        return city_coords[city_name]
    
    # Match with base name
    if base_city in city_coords:
        return city_coords[base_city]
    
    # If no match found, return None so we can filter it out
    return None

# Function to check if a city exists in coordinates
def city_has_coordinates(city_name):
    return get_coordinates_for_city(city_name) is not None

# Haversine distance calculator
def haversine_distance(lat1, lon1, lat2, lon2):
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

# Function to create points along an arc path
def create_arc_path(start_coords, end_coords, steps=20):
    points = []
    # Linear interpolation with altitude variation
    for i in range(steps + 1):
        t = i / steps
        # Linear interpolation for lat/lng
        lat = start_coords[0] + t * (end_coords[0] - start_coords[0])
        lng = start_coords[1] + t * (end_coords[1] - start_coords[1])
        
        # Parabolic interpolation for altitude (highest in the middle)
        altitude = 0
        if i > 0 and i < steps:
            # Create a parabola peaking in the middle
            normalized_t = (t - 0.5) * 2  # -1 to 1
            altitude = 100000 * (1 - normalized_t**2)  # Higher in the middle
        
        points.append({"position": [lng, lat, altitude], "t": t})
    
    return points

# 3D flight route visualization with PyDeck
def create_3d_flight_map(departure_city, arrival_city):
    # Get coordinates
    dep_coords = get_coordinates_for_city(departure_city)
    arr_coords = get_coordinates_for_city(arrival_city)
    
    # Create a DataFrame with the flight path
    # We'll create multiple points along the path for the arc
    path_points = create_arc_path(dep_coords, arr_coords)
    
    # Create arc layer for the flight path
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=[{
            "source": dep_coords,
            "target": arr_coords,
            "source_name": departure_city,
            "target_name": arrival_city
        }],
        get_source_position=["source[1]", "source[0]"],  # [lng, lat]
        get_target_position=["target[1]", "target[0]"],  # [lng, lat]
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
    
    # Create a column layer to show elevation at the cities
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=[
            {"position": [dep_coords[1], dep_coords[0]], "name": departure_city, "elevation": 40000, "color": [0, 255, 0, 180]},
            {"position": [arr_coords[1], arr_coords[0]], "name": arrival_city, "elevation": 40000, "color": [255, 0, 0, 180]}
        ],
        get_position="position",
        get_elevation="elevation",
        get_fill_color="color",
        radius=10000,
        pickable=True,
        auto_highlight=True,
    )
    
    # Create a set of smaller points to represent other airports (optional)
    # This adds more context to the map
    other_airports = []
    for city, coords in city_coords.items():
        if city not in [departure_city, arrival_city]:
            other_airports.append({
                "position": [coords[1], coords[0]], 
                "name": city, 
                "color": [100, 100, 100, 120]
            })
    
    other_airports_layer = pdk.Layer(
        "ScatterplotLayer",
        data=other_airports,
        get_position="position",
        get_radius=5000,
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
        layers=[arc_layer, column_layer, scatter_layer, other_airports_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10"  # Dark theme for better 3D visualization
    )
    
    return r

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
        airlines = top_airlines  # Use the top 5 airlines we identified
        
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
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
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
        # Airline selection - now restricted to top 5
        airline = st.selectbox("Select Airline", options=airlines)
    
    # Get valid departure cities for this airline (cities with known coordinates)
    valid_departure_cities = df[df["Airline"] == airline]["Departure City"].unique().tolist()
    
    with col2:
        # Departure city selection
        departure_city = st.selectbox("Departure City", options=valid_departure_cities)
    
    # Get valid arrival cities for this airline and departure city (cities with known coordinates)
    valid_arrival_cities = df[(df["Airline"] == airline) & 
                             (df["Departure City"] == departure_city)]["Arrival City"].unique().tolist()
    
    # If no valid arrival cities, use all arrival cities for this airline
    if not valid_arrival_cities:
        valid_arrival_cities = df[df["Airline"] == airline]["Arrival City"].unique().tolist()
    
    with col3:
        # Arrival city selection - use only valid arrival cities
        arrival_city = st.selectbox(
            "Arrival City", 
            options=valid_arrival_cities,
            index=min(0, len(valid_arrival_cities)-1)
        )
    
    # Predict button in its own row
    if st.button("Predict Ticket Price & Show Route", type="primary"):
        try:
            # Create columns for results display
            map_col, info_col = st.columns([3, 2])
            
            with map_col:
                st.subheader("Flight Route (3D View)")
                # Enable print statements for debugging
                sys.stdout = original_stdout
                
                # Create and display 3D map with PyDeck
                flight_map = create_3d_flight_map(departure_city, arrival_city)
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
                    # Otherwise use model to predict
                    # Convert selections to encoded values
                    airline_encoded = label_encoders["Airline"].transform([airline])[0]
                    departure_city_encoded = label_encoders["Departure City"].transform([departure_city])[0]
                    arrival_city_encoded = label_encoders["Arrival City"].transform([arrival_city])[0]
                    
                    # Calculate direct distance based on coordinates
                    dep_coords = get_coordinates_for_city(departure_city)
                    arr_coords = get_coordinates_for_city(arrival_city)
                    
                    # Calculate direct distance
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
                    st.markdown(f"""
                    * **Airline**: {airline}
                    * **Route**: {departure_city} to {arrival_city}
                    * **Distance**: {distance:.0f} miles
                    * **Price**: {'$' + f"{avg_fare:.2f}" if 'avg_fare' in locals() else '$' + f"{predicted_price:.2f}"}
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
                    if st.button("Predict Ticket Price & Show Route", type="primary"):
                        # After your existing visualizations, add a new section for advanced analytics
                        st.markdown("---")
                        st.header("🔍 Advanced Flight Analytics")
        
                        # Create tabs for different visualization types
                        analytics_tab1, analytics_tab2 = st.tabs(["Price History & Insights", "Data Visualizations"])
        
                    with analytics_tab1:
                        col1, col2 = st.columns(2)
            
                    with col1:
                    # Add price history chart
                        price_history_fig = create_price_history_chart(df, airline, departure_city, arrival_city)
                        st.plotly_chart(price_history_fig, use_container_width=True)
            
                    with col2:
                    # Add best booking time visualization
                        add_best_booking_time(df, airline, departure_city, arrival_city)
            
                    # Add seasonal pricing insights
                        add_seasonal_pricing_insight(df)
        
                    with analytics_tab2:
                    # Show all the additional data visualizations
                        show_additional_visualizations(df, airline, departure_city, arrival_city, get_coordinates_for_city)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            # Print the full exception traceback for debugging
            import traceback
            st.write(traceback.format_exc())

    if __name__ == "__main__":
        main()