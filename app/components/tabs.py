import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from maps import (  
    create_price_heatmap, create_travel_time_map,
    create_price_comparison_chart, create_price_time_series
)
from utils import get_quartier_statistics, get_price_history

def create_tabs(df_quartier, df_travel_times, quartier_coords, selected_quartier, 
                selected_zimmer, selected_baujahr, predicted_price, travel_times, 
                quartier_options, apply_chart_styling):
    """Creates all tab content for the application"""
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Immobilienanalyse", 
        "🗺️ Standort", 
        "📈 Marktentwicklungen",
        "🧠 Machine-Learning-Modell"
    ])
    
    # Tab 1: Property Analysis
    with tab1:
        property_analysis_tab(
            df_quartier, 
            selected_quartier, 
            predicted_price, 
            travel_times, 
            apply_chart_styling
        )
    
    # Tab 2: Location
    with tab2:
        location_tab(
            df_quartier, 
            df_travel_times, 
            quartier_coords,
            apply_chart_styling
        )
    
    # Tab 3: Market Trends
    with tab3:
        market_trends_tab(
            df_quartier, 
            quartier_options, 
            selected_quartier, 
            selected_zimmer, 
            apply_chart_styling
        )
    
    # Tab 4: ML Model
    with tab4:
        ml_model_tab(
            df_quartier, 
            selected_quartier, 
            selected_zimmer, 
            apply_chart_styling
        )

def property_analysis_tab(df_quartier, selected_quartier, predicted_price, travel_times, apply_chart_styling):
    """Content for the Property Analysis tab"""
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
                title=f"Reisezeiten ab {selected_quartier}"
            )
            
            apply_chart_styling(fig)
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
        
        apply_chart_styling(fig)
        fig.update_layout(
            yaxis_title="Medianpreis (CHF)",
            xaxis_title="Jahr"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Für dieses Viertel sind keine historischen Preisdaten verfügbar.")

# Define the other tab functions (location_tab, market_trends_tab, ml_model_tab) 
# similar to property_analysis_tab, extracting the code from the original app.py