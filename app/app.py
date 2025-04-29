"""
Main Application for Zurich Real Estate Price Prediction
-------------------------------------------------------
Purpose: Streamlit web application for price prediction and visualization

Tasks:
1. Create user interface for inputting property details
2. Display predicted prices based on user input
3. Visualize property prices and travel times on interactive maps
4. Show data insights and model explanations

Owner: Matteo (Primary), Anna (Support)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import sys
from datetime import datetime

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from app.utils import (
    load_model, load_datasets, prepare_prediction_input, predict_price,
    format_price, get_price_range, get_neighborhoods, get_neighborhood_coordinates,
    get_key_destinations, get_building_age_options, get_room_count_options,
    convert_age_category_to_years
)
from app.maps import (
    create_price_heatmap, create_travel_time_map, create_combined_map, create_interactive_map
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Zurich Real Estate Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    """Load model and datasets."""
    # Load model
    model, features = load_model()
    
    # Load datasets
    neighborhood_df, building_age_df, travel_time_df = load_datasets()
    
    # Get coordinates
    neighborhood_coords = get_neighborhood_coordinates()
    key_destinations = get_key_destinations()
    
    return model, features, neighborhood_df, building_age_df, travel_time_df, neighborhood_coords, key_destinations

def create_sidebar():
    """Create sidebar with user inputs."""
    st.sidebar.title("Property Details")
    
    # Neighborhood selection
    neighborhoods = get_neighborhoods()
    neighborhood = st.sidebar.selectbox(
        "Neighborhood",
        neighborhoods,
        index=0
    )
    
    # Room count selection
    room_options = get_room_count_options()
    room_count = st.sidebar.selectbox(
        "Number of Rooms",
        room_options,
        index=4  # Default to 3 rooms
    )
    
    # Building age selection
    age_options = get_building_age_options()
    building_age_category = st.sidebar.selectbox(
        "Building Age",
        age_options,
        index=6  # Default to 1991-2000
    )
    
    # Convert building age category to years
    building_age = convert_age_category_to_years(building_age_category)
    
    # Travel time preferences
    st.sidebar.subheader("Travel Time Preferences")
    
    max_travel_time_hauptbahnhof = st.sidebar.slider(
        "Max travel time to Hauptbahnhof (min)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    max_travel_time_eth = st.sidebar.slider(
        "Max travel time to ETH Zurich (min)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    max_travel_time_airport = st.sidebar.slider(
        "Max travel time to Zurich Airport (min)",
        min_value=5,
        max_value=90,
        value=45,
        step=5
    )
    
    max_travel_time_bahnhofstrasse = st.sidebar.slider(
        "Max travel time to Bahnhofstrasse (min)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    # Collect travel time preferences
    travel_times = {
        "Hauptbahnhof": max_travel_time_hauptbahnhof,
        "ETH_Zurich": max_travel_time_eth,
        "Zurich_Airport": max_travel_time_airport,
        "Bahnhofstrasse": max_travel_time_bahnhofstrasse
    }
    
    # Add a calculate button
    calculate_button = st.sidebar.button("Calculate Price")
    
    return neighborhood, room_count, building_age, travel_times, calculate_button

def display_prediction(model, features, neighborhood, room_count, building_age, travel_times):
    """Display price prediction."""
    # Prepare input data
    input_data = prepare_prediction_input(neighborhood, room_count, building_age, travel_times)
    
    # Make prediction
    predicted_price = predict_price(model, features, input_data)
    
    # Display prediction
    st.header("Property Price Prediction")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Price",
            value=format_price(predicted_price)
        )
    
    with col2:
        low_range, high_range = get_price_range(predicted_price)
        st.metric(
            label="Price Range (±10%)",
            value=f"{low_range} - {high_range}"
        )
    
    with col3:
        price_per_room = predicted_price / room_count if predicted_price and room_count else None
        st.metric(
            label="Price per Room",
            value=format_price(price_per_room)
        )
    
    st.write("""
    **Note**: This prediction is based on historical data and the characteristics you provided.
    Actual market prices may vary based on additional factors like property condition, specific location,
    amenities, and current market conditions.
    """)

def display_maps(neighborhood_df, travel_time_df, neighborhood_coords, key_destinations, neighborhood, travel_times):
    """Display interactive maps."""
    st.header("Interactive Zurich Maps")
    
    # Create tabs for different map views
    tab1, tab2, tab3 = st.tabs(["Price Map", "Travel Time", "Combined View"])
    
    with tab1:
        st.subheader("Property Prices across Zurich")
        price_map = create_price_heatmap(neighborhood_df, neighborhood_coords)
        st.plotly_chart(price_map, use_container_width=True)
    
    with tab2:
        st.subheader("Travel Times from Key Destinations")
        
        # Create selectbox for different destinations
        destinations = list(key_destinations.keys())
        selected_destination = st.selectbox(
            "Select Destination",
            destinations,
            index=0
        )
        
        # Create travel time map
        travel_map = create_travel_time_map(
            travel_time_df,
            neighborhood_coords,
            key_destinations[selected_destination],
            selected_destination.replace("_", " ")
        )
        st.plotly_chart(travel_map, use_container_width=True)
    
    with tab3:
        st.subheader("Interactive Property Map")
        
        # Create interactive map with filters
        interactive_map = create_interactive_map(
            neighborhood_df,
            travel_time_df,
            neighborhood_coords,
            key_destinations,
            selected_neighborhood=neighborhood,
            max_travel_time=travel_times["Hauptbahnhof"]
        )
        st.plotly_chart(interactive_map, use_container_width=True)
    
    st.write("""
    **Map Legend**:
    - Property price levels are shown in the color scale, with higher prices in darker colors
    - Travel times are shown in minutes, with shorter times in greener colors
    - The blue circle shows the approximate travel time radius from the selected neighborhood
    - Red markers indicate key destinations in Zurich
    """)

def display_insights(neighborhood_df, building_age_df):
    """Display data insights and model explanation."""
    st.header("Market Insights")
    
    # Create tabs for different insights
    tab1, tab2, tab3 = st.tabs(["Price Trends", "Neighborhood Comparison", "Building Age Impact"])
    
    with tab1:
        st.subheader("Price Trends Over Time")
        st.write("""
        This section will show historical price trends from 2009-2024.
        Charts will be implemented with actual data once available.
        """)
        
        # Placeholder for price trend chart
        st.info("Price trend chart will be displayed here.")
    
    with tab2:
        st.subheader("Neighborhood Price Comparison")
        st.write("""
        This section will compare prices across different Zurich neighborhoods.
        Charts will be implemented with actual data once available.
        """)
        
        # Placeholder for neighborhood comparison chart
        st.info("Neighborhood comparison chart will be displayed here.")
    
    with tab3:
        st.subheader("Impact of Building Age")
        st.write("""
        This section will show how building age affects property prices.
        Charts will be implemented with actual data once available.
        """)
        
        # Placeholder for building age impact chart
        st.info("Building age impact chart will be displayed here.")

def main():
    """Main application function."""
    # Set app title
    st.title("🏡 Zurich Real Estate Price Prediction")
    
    # Add app description
    st.write("""
    This application predicts real estate prices in Zurich based on neighborhood,
    room count, building age, and travel time to key destinations.
    
    Use the sidebar to input property details and see the predicted price,
    along with interactive maps and market insights.
    """)
    
    try:
        # Load data
        model, features, neighborhood_df, building_age_df, travel_time_df, neighborhood_coords, key_destinations = load_data()
        
        # Create sidebar with inputs
        neighborhood, room_count, building_age, travel_times, calculate_button = create_sidebar()
        
        # Display prediction if calculate button is clicked
        if calculate_button:
            display_prediction(model, features, neighborhood, room_count, building_age, travel_times)
        
        # Display maps
        display_maps(neighborhood_df, travel_time_df, neighborhood_coords, key_destinations, neighborhood, travel_times)
        
        # Display insights
        display_insights(neighborhood_df, building_age_df)
        
        # Add footer with about section
        st.markdown("---")
        st.markdown("""
        **About this app**:  
        Developed as part of the Zurich Real Estate Price Prediction project.  
        Data sources: Property Prices by Neighborhood (bau515od5155.csv), Property Prices by Building Age (bau515od5156.csv), and Google Maps API for travel times.
        """)
    
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure all required files are available.")
        logger.error(f"File not found: {e}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error in main application: {e}")

if __name__ == "__main__":
    main()
