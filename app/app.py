import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import sys

# Add script directory to path
sys.path.append('./scripts')
sys.path.append('./app')

# Import helper modules
from utils import load_data, preprocess_data
from maps import create_price_heatmap, create_travel_time_map

# Set page configuration
st.set_page_config(
    page_title="Zurich Real Estate Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏡 Zurich Real Estate Price Prediction")
st.markdown("""
This app predicts real estate prices in Zurich based on property characteristics and travel time.
Select parameters on the left sidebar to get price predictions and view visualizations.
""")

# Cache data loading
@st.cache_data
def get_data():
    # Load the neighborhood dataset
    neighborhood_data = load_data("data/raw/bau515od5155.csv")
    # Load the building age dataset
    building_age_data = load_data("data/raw/bau515od5156.csv")
    
    # Process and return the data
    return preprocess_data(neighborhood_data, building_age_data)

# Try to load data
try:
    data, neighborhoods, room_counts, building_ages, latest_year = get_data()
    
    # Sidebar inputs
    st.sidebar.header("Property Parameters")
    
    selected_neighborhood = st.sidebar.selectbox(
        "Select Neighborhood",
        options=neighborhoods,
        index=0
    )
    
    selected_rooms = st.sidebar.selectbox(
        "Number of Rooms",
        options=room_counts,
        index=2  # Default to 3-4 rooms
    )
    
    selected_building_age = st.sidebar.selectbox(
        "Building Age",
        options=building_ages,
        index=len(building_ages) // 2  # Default to middle age range
    )
    
    # Travel time preferences
    st.sidebar.header("Travel Time Preferences")
    max_travel_time = st.sidebar.slider(
        "Maximum Travel Time (minutes)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    key_destinations = ["Hauptbahnhof", "ETH Zurich", "Zurich Airport", "Bahnhofstrasse"]
    selected_destinations = st.sidebar.multiselect(
        "Key Destinations",
        options=key_destinations,
        default=["Hauptbahnhof"]
    )
    
    # Main content area - split into columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Price Prediction")
        
        # Load model if available, otherwise display placeholder
        model_path = os.path.join("models", "price_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                
            # Create input features for prediction
            features = {
                'neighborhood': selected_neighborhood,
                'room_count': selected_rooms,
                'building_age': selected_building_age,
                'travel_time': max_travel_time
            }
            
            # Make prediction (placeholder for now)
            # In reality, we would transform these features properly
            predicted_price = 1000000  # Placeholder
            
            st.metric(
                label="Estimated Price (CHF)",
                value=f"{predicted_price:,.0f}"
            )
            
            st.info(f"This estimate is based on {latest_year} data for a {selected_rooms} room property in {selected_neighborhood} built during {selected_building_age}.")
        else:
            st.warning("Price prediction model not found. Please train the model first.")
            st.info("Using historical data to show neighborhood averages instead.")
            
            # Filter data for visualization
            filtered_data = data[
                (data['neighborhood'] == selected_neighborhood) & 
                (data['room_count'] == selected_rooms)
            ].sort_values('year')
            
            if not filtered_data.empty:
                st.line_chart(filtered_data.set_index('year')['median_price'])
            else:
                st.error("No matching data found for the selected parameters.")
    
    with col2:
        st.header("Price Distribution Map")
        
        # Create map visualization
        map_data = data[data['year'] == latest_year]
        st.plotly_chart(create_price_heatmap(map_data))
    
    # Travel time analysis section
    st.header("Travel Time Analysis")
    st.markdown("""
    Note: Travel time data is currently a placeholder. In the complete app, this would show
    actual travel times from the selected neighborhood to key destinations.
    """)
    
    # Placeholder for travel time visualization
    travel_time_placeholder = pd.DataFrame({
        'Destination': key_destinations,
        'Travel Time (min)': [25, 18, 35, 20]
    })
    
    st.bar_chart(travel_time_placeholder.set_index('Destination'))
    
    # Historical price trends
    st.header("Historical Price Trends")
    
    # Filter for the selected neighborhood
    neighborhood_history = data[data['neighborhood'] == selected_neighborhood]
    
    if not neighborhood_history.empty:
        fig = px.line(
            neighborhood_history,
            x='year',
            y='median_price',
            color='room_count',
            title=f"Price History in {selected_neighborhood}"
        )
        st.plotly_chart(fig)
    else:
        st.error("No historical data available for the selected neighborhood.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please make sure the CSV files are in the correct location: data/raw/")