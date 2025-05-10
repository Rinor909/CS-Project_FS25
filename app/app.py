import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

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
    
    # Page configuration - clean and wide layout
    st.set_page_config(
        page_title="ImmoInsight ZH",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
        # Logo area
        st.image("https://i.ibb.co/Fb2X2QRB/Logo-Immo-Insight-ZH-w-bg.png", width=300)
    
    with header_col2:
        # Title and subtitle
        st.title("Zürcher Immobilien. Datenbasiert. Klar.")
        st.caption("Immobilienpreise in Zürich datengetrieben prognostizieren.")
    
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
            st.subheader("Immobilienstandort")
            selected_quartier = st.selectbox(
                "Der Immobilienstandort geht hier hin:",
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
            st.subheader("Zimmeranzahl")
            selected_zimmer = st.slider(
                "",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                format="%d Zimmer"
            )
            
                # Construction year with dropdown
            st.subheader("Baujahr")
            selected_baujahr = st.selectbox(
                "",
                options=list(range(1900, 2026, 5)),
                index=20,  # Default to 2000
                format_func=lambda x: str(x),
            )
            # Transportation mode - Put this in its own container
        with st.container(border=True):
            st.subheader("Transportmittel")
            selected_transport = st.radio(
                "",
                options=["öffentlicher Verkehr", "Auto"],
                horizontal=True,
                index=0
            )
        # Map selection values back to original keys
        selected_transport = "transit" if selected_transport == "Public Transit" else "driving"
        
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
            label="Geschätzer Immobilienwert",
            value=f"{predicted_price:,.0f} CHF" if predicted_price else "N/A",
            delta=f"{round((predicted_price / 1000000 - 1) * 100, 1):+.1f}%" if predicted_price else None,
            delta_color="inverse"
        )
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "📊 Immobilienanalyse", 
            "🗺️ Standort", 
            "📈 Marktentwicklungen"
        ])
        
        # Tab 1: Property Analysis
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Neighborhood statistics
                st.subheader("Nachbarschaftsstatistiken")
                
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
                    {"Reiseziel": key, "Minuten": value} for key, value in travel_times.items()
                ]
                df_travel_viz = pd.DataFrame(travel_times_data)

                if not df_travel_viz.empty:
                    fig = px.bar(
                        df_travel_viz,
                        x="Reiseziel",
                        y="Minuten",
                        #color="Minutes",
                        #color_continuous_scale="Blues",
                        title=f"Reisezeiten ab {selected_quartier}"
                    )
                    
                    # Improve figure styling
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(family="Arial, sans-serif", size=12),
                        margin=dict(l=40, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Für dieses Viertel sind keine Reisezeitdaten verfügbar.")
            
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
                    color_discrete_sequence=["#1565C0"]
                )
                
                # Improve figure styling
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=12),
                    margin=dict(l=40, r=20, t=40, b=20),
                    yaxis_title="Medianpreis (CHF)",
                    xaxis_title="Jahr"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Für dieses Viertel sind keine historischen Preisdaten verfügbar.")
        
        # Tab 2: Location
    with tab2:
        st.subheader("Interaktive Karten")
        
        # Map type selection
        map_type = st.radio(
            "Map Type",
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
                
                # Let the map function handle layout - don't override here
                st.plotly_chart(price_map, use_container_width=True)
            except Exception as e:
                st.error(f"Fehler beim Erstellen der Preistabelle: {str(e)}")
                st.info("Stellen Sie sicher, dass Sie ordnungsgemäss verarbeitete Daten und gültige Koordinaten haben.")
            
        else:  # Travel Times
            # Selection for destination and transport mode
            col1, col2 = st.columns(2)
            
            with col1:
                zielorte = df_travel_times['Zielort'].unique() if not df_travel_times.empty and 'Zielort' in df_travel_times.columns else ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']
                selected_ziel = st.selectbox("Zielort", options=zielorte, index=0)
            
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
                
                # Let the map function handle layout - don't override here
                st.plotly_chart(travel_map, use_container_width=True)
            except Exception as e:
                st.error(f"Fehler beim Erstellen der Reisekarte: {str(e)}")
                st.info("Stellen Sie sicher, dass Sie zunächst Reisezeitdaten generiert haben.")
        
        # Tab 3: Market Trends
        with tab3:
            st.subheader("Nachbarschaftsvergleich")
            
            # Select multiple neighborhoods for comparison
            compare_quartiere = st.multiselect(
                "Wählen Sie Nachbarschaften zum Vergleich",
                options=quartier_options,
                default=[selected_quartier]
            )
            
            if len(compare_quartiere) > 0:
                # Select number of rooms for comparison
                compare_zimmer = st.select_slider(
                    "Zimmeranzahl zum Vergleich",
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
                    
                    # Time series comparison
                    st.subheader("Preis Trends im Vergleich")
                    
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
                    st.error(f"Fehler beim Erstellen der Vergleichskarten: {str(e)}")
                    st.info("Make sure you have properly processed data.")
                
                # Feature Importance
                st.subheader("Preisbeeinflussende Faktoren")
                
                # Simulated Feature Importance for the demo
                importance_data = {
                    'Feature': ['Nachbarschaft', 'Reisezeit nach HB', 'Zimmeranzahl', 'Baujahr', 'Reisezeit nach Flughafen'],
                    'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
                }
                df_importance = pd.DataFrame(importance_data)
                
                fig = px.bar(
                    df_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Faktoren, die die Immobilienpreise beeinflussen',
                    #color='Importance',
                    #color_continuous_scale='Blues'
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
                st.info("Bitte wählen Sie mindestens ein Stadtviertel zum Vergleich aus.")
    
    # ---- MAP SECTION ----
    # Create a map of Zurich
    st.subheader("Zürich Karte")
    
    # Get Zurich coordinates
    # zurich_coords = get_zurich_coordinates()
    
    # # Create a simple map using st.map
    # df_map = pd.DataFrame({
    #     'lat': [zurich_coords['latitude']],
    #     'lon': [zurich_coords['longitude']]
    # })
    
    # st.map(df_map, zoom=12, use_container_width=True)
    coords = get_zurich_coordinates()

    map_folium = folium.Map(location=[coords['latitude'], coords['longitude']], zoom_start=13, tiles='OpenStreetMap')

    folium.Marker(
        [coords['latitude'], coords['longitude']],
        tooltip="Zurich",
        popup="City (Kreis 1)"
    ).add_to(map_folium)

    st_folium(map_folium, width=1400, height=500)
    
    # ---- FOOTER ----
    st.caption("Entwickelt im Rahmen des CS-Kurses an der HSG | Datenquellen: " "[Immobilienpreise nach Quartier](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-2) | " "[Immobillienpreise nach Baualter](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-3")

if __name__ == "__main__":
    # Put our modules in the path
    import sys
    sys.path.append('app')
    
    # Run the main function
    main()