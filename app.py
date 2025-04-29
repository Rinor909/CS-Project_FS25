import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import pickle

# Add app directory to path to ensure imports work properly
sys.path.append(os.path.abspath('app'))

# Import helper modules
import sys
sys.path.append('app')  # Make sure this points to the correct directory
from utils import load_data, preprocess_data, load_travel_times, load_model
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
    """Load and preprocess the data"""
    try:
        # Load the neighborhood dataset
        neighborhood_data = load_data("data/raw/bau515od5155.csv")
        # Load the building age dataset
        building_age_data = load_data("data/raw/bau515od5156.csv")
        
        # Process and return the data
        return preprocess_data(neighborhood_data, building_age_data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, [], [], [], None

# Cache travel time data loading
@st.cache_data
def get_travel_times():
    """Load travel time data"""
    try:
        return load_travel_times("data/processed/travel_times.json")
    except Exception as e:
        st.warning(f"Travel time data not available: {str(e)}")
        return {}

# Cache model loading
@st.cache_resource
def get_model():
    """Load the trained model"""
    try:
        return load_model("models/price_model.pkl")
    except Exception as e:
        st.warning(f"Price prediction model not available: {str(e)}")
        return None

# Load data, travel times, and model
data, neighborhoods, room_counts, building_ages, latest_year = get_data()
travel_times = get_travel_times()
model = get_model()

# Main app logic
if data is not None and len(data) > 0:
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
        index=2 if len(room_counts) > 2 else 0  # Default to 3-4 rooms if available
    )
    
    selected_building_age = st.sidebar.selectbox(
        "Building Age",
        options=building_ages,
        index=len(building_ages) // 2 if building_ages else 0  # Default to middle age range
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
        
        # If model is available, make prediction
        if model is not None:
            # Prepare input features for prediction
            features = pd.DataFrame({
                'neighborhood': [selected_neighborhood],
                'room_count': [selected_rooms],
                'year': [latest_year]
            })
            
            # Add travel time if model supports it
            if travel_times and selected_neighborhood in travel_times:
                neighborhood_times = travel_times[selected_neighborhood]
                avg_travel_time = np.mean([
                    neighborhood_times.get(dest, 30) for dest in selected_destinations
                ]) if selected_destinations else 30
                
                features['avg_travel_time'] = avg_travel_time
            
            # Make prediction
            try:
                predicted_price = model.predict(features)[0]
                
                st.metric(
                    label="Estimated Price (CHF)",
                    value=f"{predicted_price:,.0f}"
                )
                
                st.info(f"This estimate is based on {latest_year} data for a {selected_rooms} room property in {selected_neighborhood}.")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Using historical data to show neighborhood averages instead.")
                predicted_price = None
        else:
            st.warning("Price prediction model not found. Using historical data instead.")
            predicted_price = None
            
        # If prediction failed or model is not available, show historical data
        if predicted_price is None:
            filtered_data = data[
                (data['neighborhood'] == selected_neighborhood) & 
                (data['room_count'] == selected_rooms)
            ].sort_values('year')
            
            if not filtered_data.empty:
                latest_data = filtered_data[filtered_data['year'] == filtered_data['year'].max()]
                if not latest_data.empty:
                    avg_price = latest_data['median_price'].mean()
                    st.metric(
                        label=f"Historical Average Price ({latest_data['year'].iloc[0]})",
                        value=f"{avg_price:,.0f} CHF"
                    )
                
                st.subheader("Historical Price Trend")
                price_chart = px.line(
                    filtered_data, 
                    x='year', 
                    y='median_price',
                    title=f"Price History for {selected_rooms} in {selected_neighborhood}"
                )
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.error("No historical data found for the selected parameters.")
    
    with col2:
        st.header("Price Distribution Map")
        
        # Create map visualization
        try:
            latest_data = data[data['year'] == latest_year]
            st.plotly_chart(create_price_heatmap(latest_data), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating map visualization: {str(e)}")
    
    # Travel time analysis section
    st.header("Travel Time Analysis")
    
    # Show travel times to selected destinations if available
    if travel_times and selected_neighborhood in travel_times:
        neighborhood_times = travel_times[selected_neighborhood]
        
        # Filter for selected destinations
        if selected_destinations:
            travel_data = {
                dest: neighborhood_times.get(dest, 0) 
                for dest in selected_destinations
            }
            
            # Create a DataFrame for visualization
            travel_df = pd.DataFrame({
                'Destination': list(travel_data.keys()),
                'Travel Time (min)': list(travel_data.values())
            })
            
            # Display as bar chart
            travel_chart = px.bar(
                travel_df, 
                x='Destination', 
                y='Travel Time (min)',
                title=f"Travel Times from {selected_neighborhood}",
                color='Travel Time (min)',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(travel_chart, use_container_width=True)
            
            # Show map of travel times
            try:
                st.plotly_chart(
                    create_travel_time_map(selected_neighborhood, selected_destinations, travel_data),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating travel time map: {str(e)}")
        else:
            st.info("Please select at least one destination to see travel times.")
    else:
        st.info("Travel time data is not available for the selected neighborhood.")
    
    # Historical price trends
    st.header("Price Analysis by Room Count")
    
    # Filter for the selected neighborhood across all room counts
    neighborhood_history = data[data['neighborhood'] == selected_neighborhood]
    
    if not neighborhood_history.empty:
        room_price_fig = px.line(
            neighborhood_history,
            x='year',
            y='median_price',
            color='room_count',
            title=f"Price History in {selected_neighborhood} by Room Count"
        )
        st.plotly_chart(room_price_fig, use_container_width=True)
        
        # Room count comparison for latest year
        latest_neighborhood = neighborhood_history[neighborhood_history['year'] == latest_year]
        if not latest_neighborhood.empty:
            room_bar_fig = px.bar(
                latest_neighborhood,
                x='room_count',
                y='median_price',
                title=f"Prices by Room Count in {selected_neighborhood} ({latest_year})",
                color='room_count'
            )
            st.plotly_chart(room_bar_fig, use_container_width=True)
    else:
        st.error("No historical data available for the selected neighborhood.")
    
    # About section in expander
    with st.expander("About this App"):
        st.markdown("""
        ## Zurich Real Estate Price Prediction App
        
        This application predicts real estate prices in Zurich based on:
        
        - **Location**: Different neighborhoods in Zurich
        - **Property Size**: Number of rooms 
        - **Building Age**: Construction period
        - **Travel Time**: Proximity to key destinations
        
        The prediction model is built using machine learning algorithms trained on historical property price data.
        
        ### Data Sources
        
        - Property prices by neighborhood (2009-2024)
        - Property prices by building age (2009-2024)
        - Generated travel time data
        
        ### Model
        
        The price prediction uses a Random Forest or Gradient Boosting model that considers location, room count, building age, and travel time factors.
        
        ### Project Team
        
        - Rinor: ML model, API integration
        - Matteo: Streamlit UI, frontend
        - Matthieu: Visualizations, maps
        - Anna: Testing, documentation
        """)
else:
    st.error("Error: Could not load the required data. Please check that the data files exist in the correct location.")
    
    # Show help for setting up the data
    st.markdown("""
    ## Troubleshooting
    
    Please make sure the required data files are available:
    
    1. Place the raw CSV files in the `data/raw/` directory:
       - `bau515od5155.csv` (Property prices by neighborhood)
       - `bau515od5156.csv` (Property prices by building age)
    
    2. Run the data preparation scripts:
       ```
       python scripts/data_preparation.py
       python scripts/generate_travel_times.py
       python scripts/model_training.py
       ```
    
    3. Restart the Streamlit app
    """)
