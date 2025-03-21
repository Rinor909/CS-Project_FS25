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
from math import radians, cos, sin, asin, sqrt

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

# Improved coordinates lookup function
def get_coordinates_for_city(city_name):
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
    
    # Handle special cases
    if "Worth" in city_name and "Dallas" in city_name:
        return city_coords.get("Dallas/Fort Worth", [32.8998, -97.0403])
    if "Fort Worth" in city_name:
        return city_coords.get("Fort Worth", [32.7555, -97.3308])
    if "Dallas" in city_name:
        return city_coords.get("Dallas", [32.7767, -96.7970])
    if "Colorado Springs" in city_name:
        return city_coords.get("Colorado Springs", [38.8339, -104.8214])
    
    # Partial matching (city is contained in known city or vice versa)
    for known_city, coords in city_coords.items():
        if base_city in known_city or known_city in base_city:
            return coords
    
    # Use coordinates for a city in the same state if possible
    state_abbr = None
    if ',' in city_name and len(city_name.split(',')) > 1:
        state_part = city_name.split(',')[1].strip()
        if len(state_part) == 2:  # State abbreviation like TX, CA, etc.
            state_abbr = state_part
            
            # Find a city in the same state
            for known_city in city_coords:
                if ',' in known_city and known_city.split(',')[1].strip() == state_abbr:
                    return city_coords[known_city]
    
    # If we get here, we need to generate coordinates
    # Instead of random coordinates, use approximate US geographic center
    lat = 39.8283  # Approximate latitude for center of US
    lng = -98.5795  # Approximate longitude for center of US
    
    # Save for future use
    city_coords[city_name] = [lat, lng]
    with open(city_coords_file, 'w') as f:
        json.dump(city_coords, f)
    
    return [lat, lng]

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

# Improved flight route map
def create_flight_map(departure_city, arrival_city):
    # Get coordinates
    dep_coords = get_coordinates_for_city(departure_city)
    arr_coords = get_coordinates_for_city(arrival_city)
    
    # Calculate center and zoom level
    center_lat = (dep_coords[0] + arr_coords[0]) / 2
    center_lng = (dep_coords[1] + arr_coords[1]) / 2
    
    # Calculate distance to determine zoom
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
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_level, 
                  tiles="CartoDB positron")
    
    # Add markers with custom icons
    folium.Marker(
        dep_coords, 
        popup=departure_city, 
        tooltip=departure_city,
        icon=folium.Icon(color="green", icon="plane", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        arr_coords, 
        popup=arrival_city,
        tooltip=arrival_city,
        icon=folium.Icon(color="red", icon="plane", prefix="fa")
    ).add_to(m)
    
    # Add a curved flight path
    # Calculate an intermediate point that's above the straight line
    mid_lat = (dep_coords[0] + arr_coords[0]) / 2
    mid_lng = (dep_coords[1] + arr_coords[1]) / 2
    
    # Push the midpoint up to create a curved arc
    # Adjust the curve based on the distance
    curve_factor = min(0.15, distance / 15000)
    lat_offset = (arr_coords[1] - dep_coords[1]) * curve_factor
    mid_lat += lat_offset
    
    # Create curved line with more points for smoother curve
    quarter_lat = (dep_coords[0] + mid_lat) / 2
    quarter_lng = (dep_coords[1] + mid_lng) / 2
    three_quarter_lat = (mid_lat + arr_coords[0]) / 2
    three_quarter_lng = (mid_lng + arr_coords[1]) / 2
    
    # Create the curved flight path
    folium.PolyLine(
        locations=[
            dep_coords, 
            [quarter_lat, quarter_lng], 
            [mid_lat, mid_lng], 
            [three_quarter_lat, three_quarter_lng], 
            arr_coords
        ],
        color="blue",
        weight=3,
        opacity=0.8,
        dash_array="5",
    ).add_to(m)
    
    # Add flight direction arrow
    folium.plugins.AntPath(
        locations=[dep_coords, arr_coords],
        color="red",
        weight=2,
        opacity=0.6,
        delay=1000,
        dash_array=[10, 20],
        pulse_color="red"
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
        df.dropna(inplace=True)
        
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
                
                # Try to use folium-plugin for a richer map, fallback to standard if not available
                try:
                    from folium import plugins
                    # Add fullscreen button
                    plugins.Fullscreen().add_to(flight_map)
                except ImportError:
                    pass
                
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
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            # Print the full exception traceback for debugging
            import traceback
            st.write(traceback.format_exc())

if __name__ == "__main__":
    main()