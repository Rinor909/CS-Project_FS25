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
        page_title="ValueState Zürich",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add modern styling
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 0;
        }
        
        /* Custom header with logo */
        .header-container {
            display: flex;
            align-items: center;
            padding: 2rem 2rem 1rem 2rem;
            background-color: white;
            border-radius: 0;
            margin-bottom: 2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .logo-container {
            display: flex;
            align-items: center;
        }
        
        .logo {
            font-size: 2rem;
            color: #1565C0;
            margin-right: 0.5rem;
        }
        
        .logo-text {
            font-weight: 700;
            color: #1565C0;
            font-size: 1.3rem;
            margin-right: 1.5rem;
        }
        
        .header-text {
            display: flex;
            flex-direction: column;
        }
        
        .main-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #212121;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }
        
        .sub-tagline {
            font-size: 1rem;
            color: #616161;
            margin-top: 0;
        }
        
        /* Main content cards */
        .content-card {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        
        /* Input fields styling */
        .stSlider > div {
            padding-top: 0.5rem;
            padding-bottom: 2rem;
        }
        
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 8px;
        }
        
        /* Custom button */
        .cta-button {
            display: inline-block;
            background-color: #1565C0;
            color: white;
            padding: 0.7rem 2rem;
            font-weight: 500;
            border-radius: 50px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .cta-button:hover {
            background-color: #0D47A1;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }
        
        /* Price display */
        .price-display-container {
            background-color: #f5f9ff;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #1565C0;
        }
        
        .price-display {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1565C0;
            margin-bottom: 0.5rem;
        }
        
        .price-label {
            font-size: 0.9rem;
            color: #616161;
        }
        
        /* Statistics cards */
        .stats-container {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            flex: 1;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08);
            border: 1px solid #f0f0f0;
        }
        
        .metric-value {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1565C0;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #616161;
            margin-top: 0.3rem;
        }
        
        /* Map container */
        .map-container {
            border-radius: 10px;
            overflow: hidden;
            margin-top: 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0 0;
            gap: 0.5rem;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            color: #1565C0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #616161;
            font-size: 0.8rem;
            margin-top: 2rem;
            border-top: 1px solid #eeeeee;
        }
        
        /* Hide default hamburger menu and footer */
        #MainMenu, footer {
            visibility: hidden;
        }
        
        /* Make sidebar more modern */
        .css-1d391kg, .css-12oz5g7 {
            background-color: white;
        }
        
        /* Sidebar headers */
        .sidebar .block-container {
            padding-top: 2rem;
        }
        
        .sidebar h2 {
            font-size: 1.2rem;
            font-weight: 600;
            color: #212121;
            margin-bottom: 1.5rem;
        }

        /* General text styling */
        h3 {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

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

    # Add a custom header with logo and tagline
    st.markdown("""
    <div class="header-container">
        <div class="logo-container">
            <div class="logo">📊</div>
            <div class="logo-text">VALUESTATE<br>ZÜRICH</div>
        </div>
        <div class="header-text">
            <div class="main-header">Smarter Real Estate Decisions Start Here.</div>
            <div class="sub-tagline">Accurately predict real estate prices in Zurich using data-driven insights.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Check if data is available
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Required data not found. Please run the data preparation scripts first.")
        st.info("Run: python scripts/data_preparation.py")
        st.info("Run: python scripts/generate_travel_times.py")
        st.info("Run: python scripts/model_training.py")
        return
    
    # Add sidebar class for styling
    st.markdown('<style>div[data-testid="stSidebarContent"] { background-color: white; }</style>', unsafe_allow_html=True)

    # Sidebar for filters and inputs
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        
        # Neighborhood selection
        st.markdown("### Property Location")
        
        # Neighborhood selection with fallback if mapping is empty
        if quartier_mapping and isinstance(quartier_mapping, dict) and len(quartier_mapping) > 0:
            inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
            quartier_options = sorted(inv_quartier_mapping.keys())
        else:
            # Fallback to neighborhoods from the data
            quartier_options = sorted(df_quartier['Quartier'].unique()) if 'Quartier' in df_quartier.columns else ['Seefeld', 'City', 'Hottingen']
            inv_quartier_mapping = {name: i for i, name in enumerate(quartier_options)}
        
        selected_quartier = st.selectbox(
            "Select neighborhood",
            options=quartier_options,
            index=0,
            help="Choose the Zurich neighborhood for your property"
        )
        
        # Get quartier code with fallback
        quartier_code = inv_quartier_mapping.get(selected_quartier, 0)
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Number of rooms selection with icon
        st.markdown("### 🚪 Number of Rooms")
        selected_zimmer = st.select_slider(
            "",
            options=zimmer_options,
            value=3
        )
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Construction year with icon
        st.markdown("### 🏗️ Construction Year")
        selected_baujahr = st.slider(
            "",
            min_value=min_baujahr,
            max_value=max_baujahr,
            value=2000,
            step=5
        )
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Transportation mode with icon
        st.markdown("### 🚌 Transportation Mode")
        selected_transport = st.radio(
            "",
            options=["Public Transit", "Car"],
            index=0,
            horizontal=True
        )
        # Map selection values back to original keys
        selected_transport = "transit" if selected_transport == "Public Transit" else "driving"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Wrap main content in a card
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
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
    
    # Improved tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📊 Property Valuation", 
        "🗺️ Location Analysis", 
        "📈 Market Trends"
    ])
    
    # Tab 1: Property Valuation (formerly Price Prediction)
    with tab1:
        # Two columns for layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<h3>Property Valuation</h3>', unsafe_allow_html=True)
            
            # Modern price display
            if predicted_price:
                st.markdown(f"""
                <div class="price-display-container">
                    <div class="price-display">{predicted_price:,.0f} CHF</div>
                    <div class="price-label">Estimated market value</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3>Travel Times</h3>', unsafe_allow_html=True)
            
            # Visualize travel times
            travel_times_data = [
                {"Destination": key, "Minutes": value} for key, value in travel_times.items()
            ]
            df_travel_viz = pd.DataFrame(travel_times_data)
            
            # Travel times as bar chart with improved styling
            if not df_travel_viz.empty:
                fig = px.bar(
                    df_travel_viz,
                    x="Destination",
                    y="Minutes",
                    color="Minutes",
                    color_continuous_scale="Blues",  # Changed to match blue theme
                    title=f"Travel Times from {selected_quartier}"
                )
                
                # Improve figure styling
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=12),
                    margin=dict(l=40, r=20, t=40, b=20),
                    coloraxis_colorbar=dict(
                        title="Minutes",
                        thicknessmode="pixels", thickness=20,
                        lenmode="pixels", len=300,
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No travel time data available for this neighborhood.")
            
            # Price trend for the neighborhood
            st.markdown('<h3>Price Development</h3>', unsafe_allow_html=True)
            
            price_history = get_price_history(selected_quartier, df_quartier)
            
            if not price_history.empty:
                fig = px.line(
                    price_history,
                    x="Jahr",
                    y="MedianPreis",
                    title=f"Price Development in {selected_quartier}",
                    markers=True,
                    color_discrete_sequence=["#1565C0"]  # Match the blue theme
                )
                
                # Improve figure styling
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=12),
                    margin=dict(l=40, r=20, t=40, b=20),
                    yaxis_title="Median Price (CHF)",
                    xaxis_title="Year"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical price data available for this neighborhood.")
    
    # Tab 2: Location Analysis (formerly Maps)
    with tab2:
        st.markdown('<h3>Interactive Maps</h3>', unsafe_allow_html=True)
        
        # Map type selection with better styling
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
                
                # Update map styling to match design
                price_map.update_layout(
                    mapbox_style="light",
                    font=dict(family="Arial, sans-serif"),
                    margin=dict(l=0, r=0, t=50, b=0),
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
                
                # Update map styling to match design
                travel_map.update_layout(
                    mapbox_style="light",
                    font=dict(family="Arial, sans-serif"),
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                
                st.plotly_chart(travel_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating travel time map: {str(e)}")
                st.info("Make sure you have generated travel time data first.")
    
    # Tab 3: Market Trends (formerly Comparison & Trends)
    with tab3:
        st.markdown('<h3>Neighborhood Comparison</h3>', unsafe_allow_html=True)
        
        # Select multiple neighborhoods for comparison with better styling
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
                # Create price comparison with improved styling
                price_comparison = create_price_comparison_chart(
                    df_quartier, 
                    compare_quartiere, 
                    selected_zimmer=compare_zimmer
                )
                
                # Improve chart styling
                price_comparison.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=12),
                    margin=dict(l=40, r=20, t=50, b=20),
                )
                
                price_comparison.update_traces(
                    marker_color='#1565C0',
                )
                
                st.plotly_chart(price_comparison, use_container_width=True)
                
                # Time series comparison with improved styling
                st.markdown('<h3>Price Trends Comparison</h3>', unsafe_allow_html=True)
                
                time_series = create_price_time_series(
                    df_quartier, 
                    compare_quartiere, 
                    selected_zimmer=compare_zimmer
                )
                
                # Improve chart styling
                time_series.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=12),
                    margin=dict(l=40, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(time_series, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating comparison charts: {str(e)}")
                st.info("Make sure you have properly processed data.")
            
            # Feature Importance with improved styling
            st.markdown('<h3>Price Influencing Factors</h3>', unsafe_allow_html=True)
            
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
                color_continuous_scale='Blues'  # Changed to match theme
            )
            
            # Improve chart styling
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Arial, sans-serif", size=12),
                margin=dict(l=40, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one neighborhood for comparison.")
    
    # Close the content card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a full-width Zurich map (similar to the one in the reference image)
    st.markdown("""
    <div class="map-container" style="height: 400px; border-radius: 12px; overflow: hidden; margin: 0 -1rem;">
        <iframe 
            width="100%" 
            height="100%" 
            frameborder="0" 
            scrolling="no" 
            marginheight="0" 
            marginwidth="0" 
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d43088.17786248372!2d8.506833899999999!3d47.3774336!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x47900a08cc0e6e11%3A0x9d0a9c357e599d0!2sZ%C3%BCrich%2C%20Switzerland!5e0!3m2!1sen!2sus!4v1683837657498!5m2!1sen!2sus"
            style="border:0;" 
            allowfullscreen="" 
            loading="lazy" 
            referrerpolicy="no-referrer-when-downgrade">
        </iframe>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom footer
    st.markdown("""
    <div class="footer">
        Developed for HSG Computer Science Project | Data Source: opendata.swiss | © 2025 ValueState Zürich
    </div>
    """, unsafe_allow_html=True)
            else:
                st.warning("Price prediction could not be calculated.")
            
            # CTA Button - could be used for some action
            st.markdown(f"""
            <div class="cta-button">
                Get Detailed Report
            </div>
            """, unsafe_allow_html=True)
            
            # Neighborhood statistics with better styling
            st.markdown('<h3>Neighborhood Statistics</h3>', unsafe_allow_html=True)
            
            quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
            
            # Calculate min_max_ratio for display
            min_max_ratio = round((predicted_price / quartier_stats['median_preis'] - 1) * 100, 1) if quartier_stats['median_preis'] > 0 else 0
            
            # Display statistics in modern cards
            st.markdown(f"""
            <div class="stats-container">
                <div class="metric-card">
                    <div class="metric-value">{quartier_stats['median_preis']:,.0f} CHF</div>
                    <div class="metric-label">Median Price</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{quartier_stats['preis_pro_qm']:,.0f} CHF</div>
                    <div class="metric-label">Price per m²</div>
                </div>
            </div>
            
            <div class="stats-container" style="margin-top: 1rem;">
                <div class="metric-card">
                    <div class="metric-value" style="color: {'green' if min_max_ratio < 0 else 'red'};">{min_max_ratio:+.1f}%</div>
                    <div class="metric-label">vs. Median</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{quartier_stats['anzahl_objekte']}</div>
                    <div class="metric-label">Data Points</div>
                </div>
            </div>
            """, unsafe_allow_html=True)