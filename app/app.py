import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Main app logic
def main():
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
    
    # Define Zurich blue color - official blue from Kanton Zürich flag
    ZURICH_BLUE = "#0038A8"  # Deep blue from Zürich flag
    
    # Dark theme colors
    DARK_BG = "#121212"        # Very dark gray for backgrounds
    DARK_CARD_BG = "#1E1E1E"   # Slightly lighter gray for cards
    DARK_TEXT = "#FFFFFF"      # White text
    GRID_COLOR = "#333333"     # Dark grid lines
    
    # Page configuration - clean and wide layout
    st.set_page_config(
        page_title="ImmoInsight Zürich",
        page_icon="🦁",  # Lion emoji for Zürich
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply dark theme to Streamlit
    st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #121212;
        color: white;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        color: white;
    }
    
    /* Text color */
    body, p, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Card styling */
    [data-testid="stExpander"] {
        background-color: #1E1E1E;
        border-color: #333333;
    }
    
    /* Radio buttons, checkboxes, sliders */
    .stRadio label, .stCheckbox label {
        color: white !important;
    }
    
    /* Headers */
    .stHeader {
        background-color: transparent !important;
    }
    
    /* Containers */
    [data-testid="stVerticalBlock"] {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #0038A8;
        color: white;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #BBBBBB !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: bold;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    /* Input fields */
    input, select, textarea {
        background-color: #333333 !important;
        color: white !important;
    }
    
    /* Tabs */
    [data-testid="stTabs"] button {
        color: white !important;
        background-color: #1E1E1E !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #0038A8 !important;
        background-color: #121212 !important;
    }
    [data-testid="stTabs"] [role="tabpanel"] {
        background-color: transparent !important;
    }
    
    /* Footer */
    footer {
        display: none;
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

    # Load data and model
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Check if data is available
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Required data not found. Please run the data preparation scripts first.")
        st.info("Run: python scripts/data_preparation.py")
        st.info("Run: python scripts/generate_travel_times.py")
        st.info("Run: python scripts/model_training.py")
        return
    
    # ---- HEADER SECTION ----
    # Two-column layout for header
    header_col1, header_col2 = st.columns([1, 3])
    
    with header_col1:
        # Logo area - use your custom logo
        st.image("images/immoinsight_logo.png", width=200)
        # If you don't have the logo file locally, use a URL:
        # st.image("https://your-image-host.com/immoinsight_logo.png", width=200)
    
    with header_col2:
        # Title and subtitle
        st.title("Immobilienpreise in Zürich")
        st.caption("Datenbasierte Immobilienprognosen für fundierte Entscheidungen")
    
    # Add a separator
    st.divider()
    
    # ---- SIDEBAR FILTERS ----
    with st.sidebar:
        # Add some space at the top
        st.write("")
        
        # Neighborhood selection with fallback if mapping is empty
        if quartier_mapping and isinstance(quartier_mapping, dict) and len(quartier_mapping) > 0:
            inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
            quartier_options = sorted(inv_quartier_mapping.keys())
        else:
            # Fallback to neighborhoods from the data
            quartier_options = sorted(df_quartier['Quartier'].unique()) if 'Quartier' in df_quartier.columns else ['Seefeld', 'City', 'Hottingen']
            inv_quartier_mapping = {name: i for i, name in enumerate(quartier_options)}
        
        # Property location with container styling
        with st.container(border=True):
            st.subheader("Standort")
            selected_quartier = st.selectbox(
                "Immobilienstandort",
                options=quartier_options,
                index=0
            )
            
            # Get quartier code with fallback
            quartier_code = inv_quartier_mapping.get(selected_quartier, 0)
        
        # Add some space
        st.write("")
        
        # Property details
        with st.container(border=True):
            # Size slider
            st.subheader("Grösse")
            selected_zimmer = st.slider(
                "",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                format="%d Zimmer"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Construction year with dropdown
                st.subheader("Baujahr")
                selected_baujahr = st.selectbox(
                    "",
                    options=list(range(1900, 2026, 5)),
                    index=20,  # Default to 2000
                    format_func=lambda x: str(x)
                )
            
            with col2:
                # Room count as dropdown
                st.subheader("Zimmer")
                rooms_display = st.selectbox(
                    "",
                    options=[f"{i} Zimmer" for i in range(1, 7)],
                    index=2  # Default to 3 rooms
                )
        
        # Transportation mode
        st.write("")
        st.subheader("Transportmittel")
        selected_transport = st.radio(
            "",
            options=["ÖV", "Auto"],
            horizontal=True,
            index=0
        )
        # Map selection values back to original keys
        selected_transport = "transit" if selected_transport == "ÖV" else "driving"
        
        # "Discover" button
        st.write("")
        discover_btn = st.button("Preis Berechnen", type="primary", use_container_width=True)
        
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
    
    # ---- MAIN CONTENT ----
    # Put main content in a container
    with st.container(border=True):
        # Property valuation section
        st.subheader("Immobilienbewertung")
        
        # Price display in a colored container
        price_container = st.container(border=False)
        price_container.metric(
            label="Geschätzter Immobilienwert",
            value=f"{predicted_price:,.0f} CHF" if predicted_price else "N/A",
            delta=f"{round((predicted_price / 1000000 - 1) * 100, 1):+.1f}%" if predicted_price else None,
            delta_color="inverse"
        )
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "📊 Immobilienanalyse", 
            "🗺️ Standort", 
            "📈 Markttrends"
        ])
        
        # Tab 1: Property Analysis
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Neighborhood statistics
                st.subheader("Quartiersstatistik")
                
                quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
                
                # Calculate stats
                min_max_ratio = round((predicted_price / quartier_stats['median_preis'] - 1) * 100, 1) if quartier_stats['median_preis'] > 0 else 0
                
                # Use columns for metrics
                m1, m2 = st.columns(2)
                m1.metric("Medianpreis", f"{quartier_stats['median_preis']:,.0f} CHF")
                m2.metric("Preis pro m²", f"{quartier_stats['preis_pro_qm']:,.0f} CHF")
                
                m3, m4 = st.columns(2)
                m3.metric("vs. Median", f"{min_max_ratio:+.1f}%", delta_color="inverse")
                m4.metric("Datenpunkte", quartier_stats['anzahl_objekte'])
            
            with col2:
                # Travel times visualization
                st.subheader("Reisezeiten")
                
                travel_times_data = [
                    {"Ziel": key, "Minuten": value} for key, value in travel_times.items()
                ]
                df_travel_viz = pd.DataFrame(travel_times_data)
                
                if not df_travel_viz.empty:
                    fig = px.bar(
                        df_travel_viz,
                        x="Ziel",
                        y="Minuten",
                        color="Minuten",
                        color_continuous_scale="Blues",
                        title=f"Reisezeiten ab {selected_quartier}"
                    )
                    
                    # Apply dark theme to the chart with Zurich blue accents
                    fig.update_layout(
                        plot_bgcolor=DARK_CARD_BG,
                        paper_bgcolor=DARK_CARD_BG,
                        font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
                        margin=dict(l=40, r=20, t=40, b=20),
                        coloraxis_colorbar=dict(
                            title="Minuten",
                            titlefont=dict(color=DARK_TEXT),
                            tickfont=dict(color=DARK_TEXT)
                        ),
                        title_font=dict(color=DARK_TEXT)
                    )
                    
                    fig.update_xaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    fig.update_yaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Keine Reisezeitdaten für dieses Quartier verfügbar.")
            
            # Price history
            st.subheader("Preisentwicklung")
            
            price_history = get_price_history(selected_quartier, df_quartier)
            
            if not price_history.empty:
                fig = px.line(
                    price_history,
                    x="Jahr",
                    y="MedianPreis",
                    title=f"Preisentwicklung in {selected_quartier}",
                    markers=True,
                    color_discrete_sequence=[ZURICH_BLUE]  # Use Zurich blue
                )
                
                # Apply dark theme to the chart
                fig.update_layout(
                    plot_bgcolor=DARK_CARD_BG,
                    paper_bgcolor=DARK_CARD_BG,
                    font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
                    margin=dict(l=40, r=20, t=40, b=20),
                    yaxis_title="Medianpreis (CHF)",
                    xaxis_title="Jahr",
                    title_font=dict(color=DARK_TEXT)
                )
                
                fig.update_xaxes(
                    title_font=dict(color=DARK_TEXT),
                    tickfont=dict(color=DARK_TEXT),
                    gridcolor=GRID_COLOR,
                    zerolinecolor=GRID_COLOR
                )
                
                fig.update_yaxes(
                    title_font=dict(color=DARK_TEXT),
                    tickfont=dict(color=DARK_TEXT),
                    gridcolor=GRID_COLOR,
                    zerolinecolor=GRID_COLOR
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Keine historischen Preisdaten für dieses Quartier verfügbar.")
        
        # Tab 2: Location
        with tab2:
            st.subheader("Interaktive Karten")
            
            # Map type selection
            map_type = st.radio(
                "Kartentyp",
                options=["Immobilienpreise", "Reisezeiten"],
                horizontal=True
            )
            
            if map_type == "Immobilienpreise":
                # Selection for year and number of rooms
                col1, col2 = st.columns(2)
                
                with col1:
                    years = sorted(df_quartier['Jahr'].unique(), reverse=True) if 'Jahr' in df_quartier.columns else [2024]
                    selected_year = st.selectbox("Jahr", options=years, index=0)
                
                with col2:
                    zimmer_options_map = sorted(df_quartier['Zimmeranzahl_num'].unique()) if 'Zimmeranzahl_num' in df_quartier.columns else [3]
                    map_zimmer = st.selectbox("Zimmeranzahl", options=zimmer_options_map, index=2 if len(zimmer_options_map) > 2 else 0)
                
                try:
                    # Create property price map
                    price_map = create_price_heatmap(
                        df_quartier, 
                        quartier_coords, 
                        selected_year=selected_year, 
                        selected_zimmer=map_zimmer
                    )
                    
                    # Update map styling for dark theme
                    price_map.update_layout(
                        mapbox_style="dark",  # Dark map style
                        font=dict(family="Arial, sans-serif", color=DARK_TEXT),
                        margin=dict(l=0, r=0, t=50, b=0),
                        paper_bgcolor=DARK_CARD_BG,
                        title_font=dict(color=DARK_TEXT)
                    )
                    
                    st.plotly_chart(price_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Fehler bei der Erstellung der Preiskarte: {str(e)}")
                    st.info("Stellen Sie sicher, dass Sie alle erforderlichen Daten korrekt verarbeitet haben.")
                
            else:  # Travel Times
                # Selection for destination and transport mode
                col1, col2 = st.columns(2)
                
                with col1:
                    zielorte = df_travel_times['Zielort'].unique() if not df_travel_times.empty and 'Zielort' in df_travel_times.columns else ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']
                    selected_ziel = st.selectbox("Ziel", options=zielorte, index=0)
                
                with col2:
                    transport_options_map = df_travel_times['Transportmittel'].unique() if not df_travel_times.empty and 'Transportmittel' in df_travel_times.columns else ['transit', 'driving']
                    map_transport = st.selectbox("Transportmittel", options=transport_options_map, index=0)
                
                try:
                    # Create travel time map
                    travel_map = create_travel_time_map(
                        df_travel_times, 
                        quartier_coords, 
                        zielort=selected_ziel, 
                        transportmittel=map_transport
                    )
                    
                    # Update map styling for dark theme
                    travel_map.update_layout(
                        mapbox_style="dark",  # Dark map style
                        font=dict(family="Arial, sans-serif", color=DARK_TEXT),
                        margin=dict(l=0, r=0, t=50, b=0),
                        paper_bgcolor=DARK_CARD_BG,
                        title_font=dict(color=DARK_TEXT)
                    )
                    
                    st.plotly_chart(travel_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Fehler bei der Erstellung der Reisezeitkarte: {str(e)}")
                    st.info("Stellen Sie sicher, dass Sie die Reisezeitdaten generiert haben.")
        
        # Tab 3: Market Trends
        with tab3:
            st.subheader("Quartiervergleich")
            
            # Select multiple neighborhoods for comparison
            compare_quartiere = st.multiselect(
                "Quartiere zum Vergleich auswählen",
                options=quartier_options,
                default=[selected_quartier]
            )
            
            if len(compare_quartiere) > 0:
                # Select number of rooms for comparison
                compare_zimmer = st.select_slider(
                    "Zimmeranzahl für den Vergleich",
                    options=[1, 2, 3, 4, 5, 6],
                    value=selected_zimmer
                )
                
                try:
                    # Create price comparison
                    price_comparison = create_price_comparison_chart(
                        df_quartier, 
                        compare_quartiere, 
                        selected_zimmer=compare_zimmer
                    )
                    
                    # Apply dark theme to the chart
                    price_comparison.update_layout(
                        plot_bgcolor=DARK_CARD_BG,
                        paper_bgcolor=DARK_CARD_BG,
                        font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
                        margin=dict(l=40, r=20, t=50, b=20),
                        title_font=dict(color=DARK_TEXT)
                    )
                    
                    # Use Zurich blue for bars
                    price_comparison.update_traces(
                        marker_color=ZURICH_BLUE,
                    )
                    
                    price_comparison.update_xaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    price_comparison.update_yaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    st.plotly_chart(price_comparison, use_container_width=True)
                    
                    # Time series comparison
                    st.subheader("Preisentwicklung im Vergleich")
                    
                    time_series = create_price_time_series(
                        df_quartier, 
                        compare_quartiere, 
                        selected_zimmer=compare_zimmer
                    )
                    
                    # Apply dark theme to the chart
                    time_series.update_layout(
                        plot_bgcolor=DARK_CARD_BG,
                        paper_bgcolor=DARK_CARD_BG,
                        font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
                        margin=dict(l=40, r=20, t=50, b=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color=DARK_TEXT)
                        ),
                        title_font=dict(color=DARK_TEXT)
                    )
                    
                    time_series.update_xaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    time_series.update_yaxes(
                        title_font=dict(color=DARK_TEXT),
                        tickfont=dict(color=DARK_TEXT),
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    )
                    
                    st.plotly_chart(time_series, use_container_width=True)
                except Exception as e:
                    st.error(f"Fehler bei der Erstellung der Vergleichsdiagramme: {str(e)}")
                    st.info("Stellen Sie sicher, dass Sie alle erforderlichen Daten korrekt verarbeitet haben.")
                
                # Feature Importance
                st.subheader("Preisbeeinflussende Faktoren")
                
                # Simulated Feature Importance for the demo
                importance_data = {
                    'Feature': ['Quartier', 'Reisezeit zum HB', 'Zimmeranzahl', 'Baujahr', 'Reisezeit zum Flughafen'],
                    'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
                }
                df_importance = pd.DataFrame(importance_data)
                
                fig = px.bar(
                    df_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Faktoren, die die Immobilienpreise beeinflussen',
                    color='Importance',
                    color_continuous_scale=['#0038A8', '#5588FF']  # Zurich blue shades
                )
                
                # Apply dark theme to the chart
                fig.update_layout(
                    plot_bgcolor=DARK_CARD_BG,
                    paper_bgcolor=DARK_CARD_BG,
                    font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
                    margin=dict(l=40, r=20, t=50, b=20),
                    coloraxis_showscale=False,
                    title_font=dict(color=DARK_TEXT)
                )
                
                fig.update_xaxes(
                    title_font=dict(color=DARK_TEXT),
                    tickfont=dict(color=DARK_TEXT),
                    gridcolor=GRID_COLOR,
                    zerolinecolor=GRID_COLOR
                )
                
                fig.update_yaxes(
                    title_font=dict(color=DARK_TEXT),
                    tickfont=dict(color=DARK_TEXT),
                    gridcolor=GRID_COLOR,
                    zerolinecolor=GRID_COLOR
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bitte wählen Sie mindestens ein Quartier für den Vergleich.")
    
    # ---- MAP SECTION ----
    # Create a map of Zurich
    st.subheader("Karte von Zürich")
    
    # Get Zurich coordinates
    zurich_coords = get_zurich_coordinates()
    
    # Create a simple map using st.map
    df_map = pd.DataFrame({
        'lat': [zurich_coords['latitude']],
        'lon': [zurich_coords['longitude']]
    })
    
    st.map(df_map, zoom=12, use_container_width=True)
    
    # ---- FOOTER ----
    st.caption("Entwickelt im Rahmen des Kurses Introduction to Computer Science an der HSG | Datenquelle: opendata.swiss | © 2025 ImmoInsight Zürich")

if __name__ == "__main__":
    # Put our modules in the path
    import sys
    sys.path.append('app')
    
    # Run the main function
    main()