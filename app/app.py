import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Eigene Module importieren
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

# Seitenkonfiguration
st.set_page_config(
    page_title="Zurich Real Estate Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS für besseres Styling
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

@st.cache_resource
def load_data_and_model():
    """Lädt alle Daten und Modelle (mit Caching für Performance)"""
    try:
        df_quartier, df_baualter, df_travel_times = load_processed_data()
        model = load_model()
        quartier_mapping = load_quartier_mapping()
        quartier_coords = get_quartier_coordinates()
        
        return df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None, None, None, None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">🏡 Zurich Real Estate Price Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    Diese App analysiert und prognostiziert Immobilienpreise in Zürich basierend auf Quartier, 
    Zimmeranzahl, Baujahr und Reisezeiten zu wichtigen Zielen.
    """)
    
    # Daten und Modell laden
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    if df_quartier is None:
        st.warning("Die Daten konnten nicht geladen werden. Bitte stellen Sie sicher, dass alle Dateien vorhanden sind.")
        return
    
    # Sidebar für Filter und Eingaben
    st.sidebar.markdown("## 🔍 Filter & Eingaben")
    
    # Quartier-Auswahl (mit inverser Mapping von Codes zu Namen)
    inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
    quartier_options = sorted(inv_quartier_mapping.keys())
    selected_quartier = st.sidebar.selectbox(
        "Quartier auswählen",
        options=quartier_options,
        index=0
    )
    quartier_code = inv_quartier_mapping[selected_quartier]
    
    # Zimmeranzahl-Auswahl
    zimmer_options = [1, 2, 3, 4, 5, 6]
    selected_zimmer = st.sidebar.select_slider(
        "Anzahl Zimmer",
        options=zimmer_options,
        value=3
    )
    
    # Baujahr-Auswahl
    min_baujahr = 1900
    max_baujahr = 2025
    selected_baujahr = st.sidebar.slider(
        "Baujahr",
        min_value=min_baujahr,
        max_value=max_baujahr,
        value=2000,
        step=5
    )
    
    # Transportmittel-Auswahl
    transport_options = ["transit", "driving"]
    selected_transport = st.sidebar.radio(
        "Transportmittel",
        options=transport_options,
        index=0,
        horizontal=True
    )
    
    # Reisezeiten für das ausgewählte Quartier abrufen
    travel_times = get_travel_times_for_quartier(
        selected_quartier, 
        df_travel_times, 
        transportmittel=selected_transport
    )
    
    # Eingaben für das Modell vorbereiten
    input_data = preprocess_input(
        quartier_code, 
        selected_zimmer, 
        selected_baujahr, 
        travel_times
    )
    
    # Preis vorhersagen
    predicted_price = predict_price(model, input_data)
    
    # Tabs für verschiedene Ansichten
    tab1, tab2, tab3 = st.tabs(["📊 Preisvorhersage", "🗺️ Karten", "📈 Vergleich & Trends"])
    
    # Tab 1: Preisvorhersage
    with tab1:
        # Zwei Spalten für Layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<div class="sub-header">Geschätzter Immobilienpreis</div>', unsafe_allow_html=True)
            
            # Preis-Display
            if predicted_price:
                st.markdown(f'<div class="price-display">{predicted_price:,.0f} CHF</div>', unsafe_allow_html=True)
            else:
                st.warning("Preisvorhersage konnte nicht berechnet werden.")
            
            # Quartier-Statistiken
            st.markdown('<div class="sub-header">Quartier-Statistiken</div>', unsafe_allow_html=True)
            
            quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
            
            # Statistiken in Cards anzeigen
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{quartier_stats['median_preis']:,.0f} CHF</div>
                    <div class="metric-label">Median-Preis</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-value">{quartier_stats['preis_pro_qm']:,.0f} CHF</div>
                    <div class="metric-label">Preis pro m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stats_col2:
                min_max_ratio = round((predicted_price / quartier_stats['median_preis'] - 1) * 100, 1)
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
                    <div class="metric-label">Datenpunkte</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="sub-header">Reisezeiten</div>', unsafe_allow_html=True)
            
            # Reisezeiten visualisieren
            travel_times_data = [
                {"Ziel": key, "Minuten": value} for key, value in travel_times.items()
            ]
            df_travel_viz = pd.DataFrame(travel_times_data)
            
            # Reisezeiten als Balkendiagramm
            if not df_travel_viz.empty:
                fig = px.bar(
                    df_travel_viz,
                    x="Ziel",
                    y="Minuten",
                    color="Minuten",
                    color_continuous_scale="Viridis_r",
                    title=f"Reisezeiten von {selected_quartier} ({selected_transport})"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Keine Reisezeit-Daten verfügbar für dieses Quartier.")
            
            # Preis-Entwicklung für das Quartier
            st.markdown('<div class="sub-header">Preisentwicklung</div>', unsafe_allow_html=True)
            
            price_history = get_price_history(selected_quartier, df_quartier)
            
            if not price_history.empty:
                fig = px.line(
                    price_history,
                    x="Jahr",
                    y="MedianPreis",
                    title=f"Preisentwicklung in {selected_quartier}",
                    markers=True
                )
                fig.update_layout(yaxis_title="Median-Preis (CHF)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Keine historischen Preisdaten für dieses Quartier verfügbar.")
    
    # Tab 2: Karten
    with tab2:
        st.markdown('<div class="sub-header">Interaktive Karten</div>', unsafe_allow_html=True)
        
        # Auswahlmöglichkeiten für Karten
        map_type = st.radio(
            "Karten-Typ",
            options=["Immobilienpreise", "Reisezeiten"],
            horizontal=True
        )
        
        if map_type == "Immobilienpreise":
            # Auswahl für Jahr und Zimmeranzahl
            col1, col2 = st.columns(2)
            
            with col1:
                years = sorted(df_quartier['Jahr'].unique(), reverse=True)
                selected_year = st.selectbox("Jahr", options=years, index=0)
            
            with col2:
                zimmer_options = sorted(df_quartier['Zimmeranzahl_num'].unique())
                map_zimmer = st.selectbox("Zimmeranzahl", options=zimmer_options, index=2)
            
            # Immobilienpreiskarte erstellen
            price_map = create_price_heatmap(
                df_quartier, 
                quartier_coords, 
                selected_year=selected_year, 
                selected_zimmer=map_zimmer
            )
            
            st.plotly_chart(price_map, use_container_width=True)
            
        else:  # Reisezeiten
            # Auswahl für Zielort und Transportmittel
            col1, col2 = st.columns(2)
            
            with col1:
                zielorte = df_travel_times['Zielort'].unique()
                selected_ziel = st.selectbox("Zielort", options=zielorte, index=0)
            
            with col2:
                transport_options = df_travel_times['Transportmittel'].unique()
                map_transport = st.selectbox("Transportmittel", options=transport_options, index=0)
            
            # Reisezeit-Karte erstellen
            travel_map = create_travel_time_map(
                df_travel_times, 
                quartier_coords, 
                zielort=selected_ziel, 
                transportmittel=map_transport
            )
            
            st.plotly_chart(travel_map, use_container_width=True)
    
    # Tab 3: Vergleich & Trends
    with tab3:
        st.markdown('<div class="sub-header">Quartier-Vergleich</div>', unsafe_allow_html=True)
        
        # Auswahl mehrerer Quartiere für den Vergleich
        compare_quartiere = st.multiselect(
            "Quartiere zum Vergleichen auswählen",
            options=quartier_options,
            default=[selected_quartier]
        )
        
        if len(compare_quartiere) > 0:
            # Zimmeranzahl für den Vergleich auswählen
            compare_zimmer = st.select_slider(
                "Anzahl Zimmer für Vergleich",
                options=zimmer_options,
                value=selected_zimmer
            )
            
            # Preisvergleich erstellen
            price_comparison = create_price_comparison_chart(
                df_quartier, 
                compare_quartiere, 
                selected_zimmer=compare_zimmer
            )
            
            st.plotly_chart(price_comparison, use_container_width=True)
            
            # Zeitreihen-Vergleich
            st.markdown('<div class="sub-header">Preistrends im Vergleich</div>', unsafe_allow_html=True)
            
            time_series = create_price_time_series(
                df_quartier, 
                compare_quartiere, 
                selected_zimmer=compare_zimmer
            )
            
            st.plotly_chart(time_series, use_container_width=True)
            
            # Feature-Importance
            st.markdown('<div class="sub-header">Einflussfaktoren auf den Preis</div>', unsafe_allow_html=True)
            
            # Simulierte Feature-Importance für die Demo
            # In einer realen Anwendung würde dies aus dem Modell extrahiert werden
            importance_data = {
                'Feature': ['Quartier', 'Reisezeit HB', 'Zimmeranzahl', 'Baujahr', 'Reisezeit Flughafen'],
                'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
            }
            df_importance = pd.DataFrame(importance_data)
            
            fig = px.bar(
                df_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Einflussfaktoren auf den Immobilienpreis',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bitte wählen Sie mindestens ein Quartier für den Vergleich aus.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Entwickelt mit ❤️ für HSG Zürich | Daten: Stadt Zürich Statistik | Letzte Aktualisierung: Mai 2025"
    )

if __name__ == "__main__":
    main()