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
    """Erstellt alle Tab-Inhalte für die Anwendung"""
    
    # Erstellt die Tab-Struktur mit Icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Immobilienanalyse",       # Grundlegende Analysen
        "🗺️ Standort",                # Kartendarstellungen
        "📈 Marktentwicklungen",       # Vergleichende Analysen
        "🧠 Machine-Learning-Modell"   # Technische Details zum Modell
    ])
    
    # Tab 1: Immobilienanalyse - Detaillierte Informationen zum ausgewählten Objekt
    with tab1:
        property_analysis_tab(
            df_quartier,            # Quartier-Datensatz
            selected_quartier,      # Ausgewähltes Quartier
            predicted_price,        # Vorhergesagter Preis
            travel_times,           # Reisezeiten
            apply_chart_styling     # Styling-Funktion
        )
    
    # Tab 2: Standort - Interaktive Karten für Preise und Reisezeiten
    with tab2:
        location_tab(
            df_quartier,            # Quartier-Datensatz
            df_travel_times,        # Reisezeit-Datensatz
            quartier_coords,        # Quartier-Koordinaten
            apply_chart_styling     # Styling-Funktion
        )
    
    # Tab 3: Marktentwicklungen - Vergleich zwischen verschiedenen Quartieren
    with tab3:
        market_trends_tab(
            df_quartier,            # Quartier-Datensatz
            quartier_options,       # Alle verfügbaren Quartiere
            selected_quartier,      # Ausgewähltes Quartier
            selected_zimmer,        # Ausgewählte Zimmeranzahl
            apply_chart_styling     # Styling-Funktion
        )
    
    # Tab 4: ML-Modell - Technische Details zum Modell
    with tab4:
        ml_model_tab(
            df_quartier,            # Quartier-Datensatz
            selected_quartier,      # Ausgewähltes Quartier
            selected_zimmer,        # Ausgewählte Zimmeranzahl
            apply_chart_styling     # Styling-Funktion
        )

def property_analysis_tab(df_quartier, selected_quartier, predicted_price, travel_times, apply_chart_styling):
    """Inhalt für den Tab Immobilienanalyse"""
    # Zweispaltiges Layout für diesen Abschnitt
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Nachbarschaftsstatistiken - Vergleichswerte für das Quartier
        st.subheader("Nachbarschaftsstatistiken")
        
        # Statistische Daten für das ausgewählte Quartier abrufen
        quartier_stats = get_quartier_statistics(selected_quartier, df_quartier)
        
        # Berechnet das Verhältnis zwischen vorhergesagtem Preis und Medianpreis des Quartiers
        min_max_ratio = round((predicted_price / quartier_stats['median_preis'] - 1) * 100, 1) if quartier_stats['median_preis'] > 0 else 0
        
        # Verwendet Spalten für Metriken - übersichtliche Darstellung
        m1, m2 = st.columns(2)
        m1.metric("Medianpreis", f"{quartier_stats['median_preis']:,.0f} CHF")  # Zeigt Medianpreis des Quartiers
        m2.metric("Preis pro m²", f"{quartier_stats['preis_pro_qm']:,.0f} CHF")  # Zeigt Quadratmeterpreis
        
        m3, m4 = st.columns(2)
        m3.metric("vs. Median", f"{min_max_ratio:+.1f}%", delta_color="inverse")  # Zeigt Prozentdifferenz zum Median
        m4.metric("Datenpunkte", quartier_stats['anzahl_objekte'])  # Zeigt Anzahl der Datenpunkte für dieses Quartier
    
    with col2:
        # Reisezeitvisualisierung - zeigt, wie lange man zu wichtigen Orten braucht
        st.subheader("Reisezeiten")
        
        # Reisezeitdaten für die Visualisierung formatieren
        travel_times_data = [
            {"Reiseziel": key, "Minuten": value} for key, value in travel_times.items()
        ]
        df_travel_viz = pd.DataFrame(travel_times_data)

        # Reisezeitdiagramm erstellen, wenn Daten verfügbar sind
        if not df_travel_viz.empty:
            # Balkendiagramm für Reisezeiten erstellen
            fig = px.bar(
                df_travel_viz,
                x="Reiseziel",                         # X-Achse: Zielorte
                y="Minuten",                           # Y-Achse: Minuten
                title=f"Reisezeiten ab {selected_quartier}"  # Dynamischer Titel mit Quartiername
            )
            
            # Einheitliches Styling anwenden
            apply_chart_styling(fig)
            # Diagramm anzeigen, volle Breite nutzen
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Informationsmeldung, wenn keine Reisezeitdaten verfügbar sind
            st.info("Für dieses Viertel sind keine Reisezeitdaten verfügbar.")
    
    # Preisentwicklung - zeigt, wie sich die Preise im Laufe der Zeit entwickelt haben
    st.subheader("Preisentwicklung")
    
    # Historische Preisdaten für das ausgewählte Quartier abrufen
    price_history = get_price_history(selected_quartier, df_quartier)
    
    # Preisentwicklungsdiagramm erstellen, wenn Daten verfügbar sind
    if not price_history.empty:
        # Liniendiagramm für historische Preisentwicklung
        fig = px.line(
            price_history,
            x="Jahr",                                # X-Achse: Jahre
            y="MedianPreis",                         # Y-Achse: Medianpreise
            title=f"Preisentwicklung in {selected_quartier}",  # Dynamischer Titel
            markers=True,                            # Marker an Datenpunkten anzeigen
            color_discrete_sequence=["#1565C0"]      # Blaue Linie
        )
        
        # Einheitliches Styling anwenden
        apply_chart_styling(fig)
        # Achsenbeschriftungen hinzufügen
        fig.update_layout(
            yaxis_title="Medianpreis (CHF)",         # Y-Achsentitel
            xaxis_title="Jahr"                       # X-Achsentitel
        )
        
        # Diagramm anzeigen, volle Breite nutzen
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Informationsmeldung, wenn keine historischen Daten verfügbar sind
        st.info("Für dieses Viertel sind keine historischen Preisdaten verfügbar.")

# Definiere die anderen Tab-Funktionen (location_tab, market_trends_tab, ml_model_tab) 
# ähnlich wie property_analysis_tab, indem der Code aus dem original app.py extrahiert wird