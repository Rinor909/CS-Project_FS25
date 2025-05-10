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
        page_title="Zurich Real Estate Price Prediction",
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
    st.markdown('<div class="main-header">🏡 Zurich Real Estate Price Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    This app analyzes and predicts real estate prices in Zurich based on neighborhood, 
    number of rooms, construction year, and travel times to important destinations.
    """)
    
    # Load data and model
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Check if data is available
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Required data not found. Please run the data preparation scripts first.")
        st.info("Run: python scripts/data_preparation.py")
        st.info("Run: python scripts/generate_travel_times.py")
        st.info("Run: python scripts/model_training.py")
        return
    
    # Sidebar for filters and inputs
    st.sidebar.markdown("## 🔍 Filters & Inputs")
    
    # Neighborhood selection with fallback if mapping is empty
    if quartier_mapping and isinstance(quartier_mapping, dict) and len(quartier_mapping) > 0:
        inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
        quartier_options = sorted(inv_quartier_mapping.keys())
    else:
        # Fallback to neighborhoods from the data
        quartier_options = sorted(df_quartier['Quartier'].unique()) if 'Quartier' in df_quartier.columns else ['Seefeld', 'City', 'Hottingen']
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
    
    # Predict price
    predicted_price = predict_price(model, input_data)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "🗺️ Maps", "📈 Comparison & Trends"])
    
    # Tab 1: Price Prediction
    with tab1:
        # Two columns for layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<div class="sub-header">Estimated Property Price</div>', unsafe_allow_html=True)
            
            # Price display
            if predicted_price:
                st.markdown(f'<div class="price-display">{predicted_price:,.0f} CHF</div>', unsafe_allow_html=True)
            else:
                st.warning("Price prediction could not be calculated.")
            
            # Neighborhood statistics
            st.markdown('<div class="sub-header">Neighborhood Statistics</div>', unsafe_allow_html=True)
            
            quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
            
            # Display statistics in cards
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{quartier_stats['median_preis']:,.0f} CHF</div>
                    <div class="metric-label">Median Price</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-value">{quartier_stats['preis_pro_qm']:,.0f} CHF</div>
                    <div class="metric-label">Price per m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stats_col2:
                min_max_ratio = round((predicted_price / quartier_stats['median_preis'] - 1) * 100, 1) if quartier_stats['median_preis'] > 0 else 0
                color = "green" if min_max_ratio < 0 else "red"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color};">{min_max_ratio:+.1f}%</div>
                    <div class="metric-label">vs. Median</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-value">{quartier_stats['anzahl_objekte']}</div>
                    <div class="metric-label">Data Points</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="sub-header">Travel Times</div>', unsafe_allow_html=True)
            
            # Visualize travel times
            travel_times_data = [
                {"Destination": key, "Minutes": value} for key, value in travel_times.items()
            ]
            df_travel_viz = pd.DataFrame(travel_times_data)
            
            # Travel times as bar chart
            if not df_travel_viz.empty:
                fig = px.bar(
                    df_travel_viz,
                    x="Destination",
                    y="Minutes",
                    color="Minutes",
                    color_continuous_scale="Viridis_r",
                    title=f"Travel Times from {selected_quartier} ({selected_transport})"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No travel time data available for this neighborhood.")
            
            # Price trend for the neighborhood
            st.markdown('<div class="sub-header">Price Development</div>', unsafe_allow_html=True)
            
            price_history = get_price_history(selected_quartier, df_quartier)
            
            if not price_history.empty:
                fig = px.line(
                    price_history,
                    x="Jahr",
                    y="MedianPreis",
                    title=f"Price Development in {selected_quartier}",
                    markers=True
                )
                fig.update_layout(yaxis_title="Median Price (CHF)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical price data available for this neighborhood.")
    
    # Tab 2: Maps
    with tab2:
        st.markdown('<div class="sub-header">Interactive Maps</div>', unsafe_allow_html=True)
        
        # Map type selection
        map_type = st.radio(
            "Map Type",
            options=["Property Prices", "Travel Times"],
            horizontal=True
        )
        
        if map_type == "Property Prices":
            # Selection for year and number of rooms
            col1, col2 = st.columns(2)
            
            with col1:
                years = sorted(df_quartier['Jahr'].unique(), reverse=True) if 'Jahr' in df_quartier.columns else [2024]
                selected_year = st.selectbox("Year", options=years, index=0)
            
            with col2:
                zimmer_options_map = sorted(df_quartier['Zimmeranzahl_num'].unique()) if 'Zimmeranzahl_num' in df_quartier.columns else [3]
                map_zimmer = st.selectbox("Number of rooms", options=zimmer_options_map, index=2 if len(zimmer_options_map) > 2 else 0)
            
            try:
                # Create property price map
                price_map = create_price_heatmap(
                    df_quartier, 
                    quartier_coords, 
                    selected_year=selected_year, 
                    selected_zimmer=map_zimmer
                )
                
                st.plotly_chart(price_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price map: {str(e)}")
                st.info("Make sure you have properly processed data and valid coordinates.")
            
        else:  # Travel Times
            # Selection for destination and transport mode
            col1, col2 = st.columns(2)
            
            with col1:
                zielorte = df_travel_times['Zielort'].unique() if not df_travel_times.empty and 'Zielort' in df_travel_times.columns else ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']
                selected_ziel = st.selectbox("Destination", options=zielorte, index=0)
            
            with col2:
                transport_options_map = df_travel_times['Transportmittel'].unique() if not df_travel_times.empty and 'Transportmittel' in df_travel_times.columns else ['transit', 'driving']
                map_transport = st.selectbox("Transport Mode", options=transport_options_map, index=0)
            
            try:
                # Create travel time map
                travel_map = create_travel_time_map(
                    df_travel_times, 
                    quartier_coords, 
                    zielort=selected_ziel, 
                    transportmittel=map_transport
                )
                
                st.plotly_chart(travel_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating travel time map: {str(e)}")
                st.info("Make sure you have generated travel time data first.")
    
    # Tab 3: Comparison & Trends
    with tab3:
        st.markdown('<div class="sub-header">Neighborhood Comparison</div>', unsafe_allow_html=True)
        
        # Select multiple neighborhoods for comparison
        compare_quartiere = st.multiselect(
            "Select neighborhoods to compare",
            options=quartier_options,
            default=[selected_quartier]
        )
        
        if len(compare_quartiere) > 0:
            # Select number of rooms for comparison
            compare_zimmer = st.select_slider(
                "Number of rooms for comparison",
                options=zimmer_options,
                value=selected_zimmer
            )
            
            try:
                # Create price comparison
                price_comparison = create_price_comparison_chart(
                    df_quartier, 
                    compare_quartiere, 
                    selected_zimmer=compare_zimmer
                )
                
                st.plotly_chart(price_comparison, use_container_width=True)
                
                # Time series comparison
                st.markdown('<div class="sub-header">Price Trends Comparison</div>', unsafe_allow_html=True)
                
                time_series = create_price_time_series(
                    df_quartier, 
                    compare_quartiere, 
                    selected_zimmer=compare_zimmer
                )
                
                st.plotly_chart(time_series, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating comparison charts: {str(e)}")
                st.info("Make sure you have properly processed data.")
            
            # Feature Importance
            st.markdown('<div class="sub-header">Price Influencing Factors</div>', unsafe_allow_html=True)
            
            # Simulated Feature Importance for the demo
            # In a real application, this would be extracted from the model
            importance_data = {
                'Feature': ['Neighborhood', 'Travel Time to HB', 'Number of Rooms', 'Construction Year', 'Travel Time to Airport'],
                'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
            }
            df_importance = pd.DataFrame(importance_data)
            
            fig = px.bar(
                df_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Factors Influencing Property Prices',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one neighborhood for comparison.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Developed with ❤️ for HSG Zurich | Data: Zurich City Statistics | Last updated: May 2025"
    )

if __name__ == "__main__":
    # Import necessary libraries at the top level
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
    
    # Put our modules in the path
    import sys
    sys.path.append('app')
    
    # Run the main function
    main()
