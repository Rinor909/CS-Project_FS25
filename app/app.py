import streamlit as st
import os
import sys

# Main app logic
def main():
    # Add this import statement at the top of the function
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    
    # Import functions from local modules
    from utils import (
        load_processed_data, load_model, load_quartier_mapping,
        preprocess_input, predict_price, get_travel_times_for_quartier,
        get_quartier_statistics, get_price_history, get_zurich_coordinates,
        get_quartier_coordinates
    )
    
    from maps import (
        create_price_heatmap, create_travel_time_map,
        create_price_comparison_chart, create_price_time_series
    )
    
    # Page configuration
    st.set_page_config(
        page_title="ImmoInsight Zürich",
        page_icon="🏡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .highlight {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .price-display {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #424242;
        }
        .debug-expander {
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Function to load data and model with caching
    @st.cache_resource
    def load_data_and_model():
        """Loads all data and models (with caching for performance)"""
        try:
            df_quartier, df_baualter, df_travel_times = load_processed_data()
            model = load_model()
            quartier_mapping = load_quartier_mapping()
            quartier_coords = get_quartier_coordinates()
            
            return df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # Return empty dataframes as fallback
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, {}, {}

    # Header
    st.markdown('<div class="main-header">🏡 ImmoInsight Zürich</div>', unsafe_allow_html=True)
    st.markdown("""
    Diese App analysiert und prognostiziert die Immobilienpreise in Zürich auf der Basis von Nachbarschaft, 
 Anzahl Zimmer, Baujahr und Fahrzeiten zu wichtigen Zielen.
    """)
    
    # Load data and model with a loading indicator
    with st.spinner("Daten und Modell werden geladen..."):
        df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Display debug info in an expander for developers
    with st.expander("Debug Info", expanded=False):
        st.write("Data Loading Status:")
        st.write(f"- Quartier data: {len(df_quartier)} records")
        st.write(f"- Baualter data: {len(df_baualter)} records")
        st.write(f"- Travel times data: {len(df_travel_times)} records")
        st.write(f"- Model loaded: {model is not None}")
        st.write(f"- Mapping loaded: {len(quartier_mapping)} neighborhoods")
    
    # Check if data is available
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Benötigte Daten nicht gefunden. Die App verwendet Standardwerte für die Darstellung.")
    
    # Sidebar for filters and inputs
    st.sidebar.markdown("## 🔍 Filters & Inputs")
    
    # Neighborhood selection with fallback if mapping is empty
    if quartier_mapping and isinstance(quartier_mapping, dict) and len(quartier_mapping) > 0:
        inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
        quartier_options = sorted(inv_quartier_mapping.keys())
    else:
        # Fallback to neighborhoods from the data
        quartier_options = sorted(df_quartier['Quartier'].unique()) if ('Quartier' in df_quartier.columns and not df_quartier.empty) else ['Seefeld', 'City', 'Hottingen']
        inv_quartier_mapping = {name: i for i, name in enumerate(quartier_options)}
    
    selected_quartier = st.sidebar.selectbox(
        "Select neighborhood",
        options=quartier_options,
        index=0
    )
    
    # Get quartier code with fallback
    quartier_code = inv_quartier_mapping.get(selected_quartier, 0)
    
    # Number of rooms selection
    zimmer_options = [1, 2, 3, 4, 5, 6]
    selected_zimmer = st.sidebar.select_slider(
        "Number of rooms",
        options=zimmer_options,
        value=3
    )
    
    # Construction year selection
    min_baujahr = 1900
    max_baujahr = 2025
    selected_baujahr = st.sidebar.slider(
        "Construction year",
        min_value=min_baujahr,
        max_value=max_baujahr,
        value=2000,
        step=5
    )
    
    # Transportation mode selection
    transport_options = ["transit", "driving"]
    selected_transport = st.sidebar.radio(
        "Transportation mode",
        options=transport_options,
        index=0,
        horizontal=True
    )
    
    # Get travel times for the selected neighborhood
    travel_times = get_travel_times_for_quartier(
        selected_quartier, 
        df_travel_times, 
        transportmittel=selected_transport
    )
    
    # Prepare inputs for the model
    input_data = preprocess_input(
        quartier_code, 
        selected_zimmer, 
        selected_baujahr, 
        travel_times
    )
    
    # Predict price with better error handling
    with st.spinner("Berechne Immobilienpreis..."):
        predicted_price = predict_price(model, input_data)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "🗺️ Maps", "📈 Comparison & Trends"])
    
    # Tab 1: Price Prediction
    with tab1:
        # Two columns for layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<div class="sub-header">Estimated Property Price</div>', unsafe_allow_html=True)
            
            # Price display with proper error handling
            if predicted_price is not None:
                st.markdown(f'<div class="price-display">{predicted_price:,.0f} CHF</div>', unsafe_allow_html=True)
            else:
                # Calculate a fallback price based on neighborhood statistics
                quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
                # Apply some basic modifiers based on inputs
                base_price = quartier_stats['median_preis']
                # Adjust for room count (more rooms = higher price)
                room_factor = 1.0 + ((selected_zimmer - 3) * 0.15)  # 15% per room difference from 3
                # Adjust for building age (newer = more expensive)
                age_factor = 1.0 + ((selected_baujahr - 1980) / 1980 * 0.3)  # Up to 30% for newest buildings
                # Calculate estimated price
                estimated_price = base_price * room_factor * age_factor
                
                st.markdown(f'<div class="price-display">{estimated_price:,.0f} CHF</div>', unsafe_allow_html=True)
                st.info("This is an estimated price based on neighborhood statistics, as the ML model prediction failed. For more accurate results, please ensure all data files are correctly loaded.")
            
            # Neighborhood statistics