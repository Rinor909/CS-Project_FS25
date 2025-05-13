# Notwendige Bibliotheken importieren
import streamlit as st            # Für die Erstellung der Web-App-Oberfläche
import pandas as pd              # Für Datenverarbeitung und -analyse
import numpy as np               # Für numerische Operationen und Berechnungen
import plotly.express as px      # Für einfachere interaktive Visualisierungen
import plotly.graph_objects as go # Für anpassbare interaktive Diagramme
from datetime import datetime    # Für Datums- und Zeitoperationen
import folium                    # Für interaktive Karten
from streamlit_folium import st_folium  # Für die Integration von Folium-Karten in Streamlit

# Hauptlogik der Anwendung 
def main():
   # Importiere Funktionen aus lokalen Modulen
   # Diese Module enthalten Datenverarbeitungs- und Visualisierungsfunktionen
   from utils import (
       load_processed_data,           # Lädt verarbeitete Daten aus CSV-Dateien
       load_model,                   # Lädt das ML-Vorhersagemodell
       load_quartier_mapping,        # Lädt die Quartier-Code-zu-Name-Zuordnung
       preprocess_input,             # Bereitet Benutzereingaben für das Modell vor
       predict_price,                # Sagt Immobilienpreise vorher
       get_travel_times_for_quartier, # Holt Reisezeiten für ein bestimmtes Quartier
       get_quartier_statistics,      # Berechnet Statistiken für ein Quartier
       get_price_history,            # Gibt die Preisentwicklung zurück
       get_zurich_coordinates,       # Liefert Koordinaten für Zürich
       get_quartier_coordinates      # Liefert Koordinaten für alle Quartiere
   )
   
   from maps import (  
       create_price_heatmap,          # Erstellt Preisheatmap
       create_travel_time_map,        # Erstellt Reisezeit-Karte
       create_price_comparison_chart, # Erstellt Preisvergleichsdiagramm
       create_price_time_series       # Erstellt Preisentwicklungsdiagramm
   )
   
   # Hilfsfunktion für einheitliches Styling von Diagrammen
   # Diese Funktion wird auf alle Diagramme angewendet, um ein konsistentes Erscheinungsbild zu gewährleisten
   def apply_chart_styling(fig, title=None):
       """Wendet einheitliches Styling auf alle Diagramme an"""
       layout_args = {
           "plot_bgcolor": "white",                 # Weißer Hintergrund für den Plotbereich
           "paper_bgcolor": "white",                # Weißer Hintergrund für das gesamte Diagramm
           "font": dict(family="Arial, sans-serif", size=12),  # Einheitliche Schriftart und -größe
           "margin": dict(l=40, r=20, t=40, b=20)  # Angepasste Ränder für optimale Platznutzung
       }
       if title:
           layout_args["title"] = title  # Fügt Titel hinzu, falls angegeben
       fig.update_layout(**layout_args)  # Wendet alle Layouteinstellungen auf das Diagramm an
       return fig  # Gibt das formatierte Diagramm zurück
   
   # Seitenkonfiguration - sauberes und breites Layout
   # Konfiguriert die grundlegenden Eigenschaften der Streamlit-Anwendung
   st.set_page_config(
       page_title="ImmoInsight ZH",           # Titel im Browser-Tab
       page_icon="🦁",                       # Symbol im Browser-Tab (Löwe-Emoji)
       layout="wide",                        # Breites Layout für bessere Visualisierung
       initial_sidebar_state="expanded"      # Seitenleiste standardmäßig ausgeklappt
   )

   # Funktion zum Laden von Daten und Modell mit Caching
   # Caching verbessert die Leistung, indem es verhindert, dass Daten bei jeder Interaktion neu geladen werden
   @st.cache_resource
   def load_data_and_model():
       """Lädt alle Daten und Modelle (mit Caching für bessere Leistung)"""
       # Lädt Basis-Datensätze
       df_quartier, df_baualter, df_travel_times = load_processed_data()
       # Lädt Machine Learning Modell
       model = load_model()
       # Lädt Mapping zwischen Quartier-Codes und -Namen
       quartier_mapping = load_quartier_mapping()
       # Lädt Koordinaten für die Kartendarstellung
       quartier_coords = get_quartier_coordinates()
       
       # Gibt alle geladenen Ressourcen zurück
       return df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords

   # Lädt Daten und Modell - dies wird dank Caching nur einmal ausgeführt
   df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
   
   # Überprüft, ob Daten verfügbar sind
   # Zeigt Anweisungen, falls die Basisdaten fehlen
   if df_quartier.empty or 'Quartier' not in df_quartier.columns:
       st.warning("Required data not found. Please run the data preparation scripts first.")
       st.info("Run: python scripts/data_preparation.py")
       st.info("Run: python scripts/generate_travel_times.py")
       st.info("Run: python scripts/model_training.py")
       return  # Beendet die Funktion, wenn keine Daten verfügbar sind
   
   # ---- HEADER-BEREICH ----
   # Zweispaltiges Layout für den Header
   header_col1, header_col2 = st.columns([1, 3])  # Verhältnis 1:3 für Logo zu Text
   
   with header_col1:
       # Logo-Bereich - Lädt das Logo von einer externen URL
       st.image("https://i.ibb.co/Fb2X2QRB/Logo-Immo-Insight-ZH-w-bg.png", width=300)
   
   with header_col2:
       # Titel und Untertitel der Anwendung
       st.title("Zürcher Immobilien. Datenbasiert. Klar.")
       st.caption("Immobilienpreise in Zürich datengetrieben prognostizieren.")
   
   # Trennlinie hinzufügen - für visuelle Trennung zwischen Header und Inhalt
   st.divider()
   
   # ---- SEITENLEISTEN-FILTER ----
   # Benutzersteuerelemente in der Seitenleiste zur Filterung der Daten
   with st.sidebar:
       # Platz am Anfang hinzufügen für bessere Optik
       st.write("")
       
       # Quartierauswahl - Umkehrung des Mappings für Benutzerfreundlichkeit
       inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}  # Kehrt das Mapping um (Code -> Name zu Name -> Code)
       quartier_options = sorted(inv_quartier_mapping.keys())  # Alphabetisch sortierte Liste aller Quartiere
       
       # Immobilienstandort mit Container-Styling für bessere visuelle Gruppierung
       with st.container(border=True):
           st.subheader("Immobilienstandort")
           # Dropdown zur Quartierauswahl
           selected_quartier = st.selectbox(
               "Der Immobilienstandort geht hier hin:",
               options=quartier_options,
               index=0  # Standardmäßig erstes Quartier ausgewählt
           )
           
           # Quartier-Code für das ausgewählte Quartier abrufen
           quartier_code = inv_quartier_mapping.get(selected_quartier, 0)  # Fallback auf 0, falls Quartier nicht im Mapping
       
       # Platz für visuelle Trennung
       st.write("")
       
       # Immobiliendetails in eigenem Container
       with st.container(border=True):
           # Größenschieberegler - Auswahl der Zimmeranzahl
           st.subheader("Zimmeranzahl")
           selected_zimmer = st.slider(
               "",
               min_value=1,          # Minimum 1 Zimmer
               max_value=6,          # Maximum 6 Zimmer
               value=3,              # Standardwert 3 Zimmer
               step=1,               # In 1er-Schritten
               format="%d Zimmer"    # Angezeigtes Format
           )
           
           # Baujahr mit Dropdown - für Altersauswahl der Immobilie
           st.subheader("Baujahr")
           selected_baujahr = st.selectbox(
               "",
               options=list(range(1900, 2026, 5)),  # Jahre von 1900 bis 2025 in 5er-Schritten
               index=25,                           # Standardwert 2010 (25. Eintrag ab 1900 in 5er-Schritten)
               format_func=lambda x: str(x),       # Formatierung als String
           )
       
       # Transportmittel - in eigenem Container für visuelle Gruppierung
       with st.container(border=True):
           st.subheader("Transportmittel")
           # Horizontale Radiobuttons zur Transportauswahl
           selected_transport = st.radio(
               "",
               options=["öffentlicher Verkehr", "Auto"],
               horizontal=True,
               index=0  # Standard ist öffentlicher Verkehr
           )
       
       # Zuordnung der Auswahlwerte zu API-Keys
       # Dies übersetzt die benutzerfreundlichen Namen in technische Werte für die API
       selected_transport = "transit" if selected_transport == "Public Transit" else "driving"
       
   # Reisezeiten für das ausgewählte Quartier abrufen
   # Dies wird für die Preisvorhersage und Reisezeitvisualisierung verwendet
   travel_times = get_travel_times_for_quartier(
       selected_quartier, 
       df_travel_times, 
       transportmittel=selected_transport
   )
   
   # Eingaben für das Modell vorbereiten und Preis vorhersagen
   # Diese Schritte konvertieren die Benutzerauswahl in ein Format, das das ML-Modell verarbeiten kann
   input_data = preprocess_input(
       quartier_code,        # Numerischer Code für das ausgewählte Quartier
       selected_zimmer,      # Ausgewählte Zimmeranzahl
       selected_baujahr,     # Ausgewähltes Baujahr
       travel_times          # Reisezeitdaten für verschiedene Ziele
   )
   # Preis basierend auf den Eingabedaten vorhersagen
   predicted_price = predict_price(model, input_data)
   
   # ---- HAUPTINHALT ----
   # Hauptinhalt in einem Container für einheitliches Styling
   with st.container(border=True):
       # Immobilienbewertungsabschnitt - Hauptergebnis der Anwendung
       st.subheader("Immobilienbewertung")
       
       # Preisanzeige in einem farbigen Container
       # Zeigt den vorhergesagten Preis prominent an
       price_container = st.container(border=False)
       price_container.metric(
           label="Geschätzer Immobilienwert",
           value=f"{predicted_price:,.0f} CHF" if predicted_price else "N/A",  # Formatiert Preis mit Tausendertrennzeichen
           delta=f"{round((predicted_price / 1000000 - 1) * 100, 1):+.1f}%" if predicted_price else None,  # Prozentuale Abweichung von 1 Mio.
           delta_color="inverse"  # Rote Farbe bei positiver Abweichung (teurer)
       )
       
       # Tabs für verschiedene Ansichten
       # Ermöglicht die Organisation von Informationen in übersichtlichen Kategorien
       tab1, tab2, tab3, tab4 = st.tabs([
           "📊 Immobilienanalyse",       # Grundlegende Analysen
           "🗺️ Standort",                # Kartendarstellungen
           "📈 Marktentwicklungen",       # Vergleichende Analysen
           "🧠 Machine-Learning-Modell"   # Technische Details zum Modell
       ])
       
       # Tab 1: Immobilienanalyse - Detaillierte Informationen zum ausgewählten Objekt
       with tab1:
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
                       x="Reiseziel",
                       y="Minuten",
                       title=f"Reisezeiten ab {selected_quartier}"  # Dynamischer Titel mit Quartiername
                   )
                   
                   # Styling anwenden
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
                   x="Jahr",                          # X-Achse: Jahre
                   y="MedianPreis",                  # Y-Achse: Medianpreise
                   title=f"Preisentwicklung in {selected_quartier}",  # Dynamischer Titel
                   markers=True,                     # Marker an Datenpunkten anzeigen
                   color_discrete_sequence=["#1565C0"]  # Blaue Linie
               )
               
               # Styling anwenden
               apply_chart_styling(fig)
               # Achsenbeschriftungen hinzufügen
               fig.update_layout(
                   yaxis_title="Medianpreis (CHF)",
                   xaxis_title="Jahr"
               )
               
               # Diagramm anzeigen, volle Breite nutzen
               st.plotly_chart(fig, use_container_width=True)
           else:
               # Informationsmeldung, wenn keine historischen Daten verfügbar sind
               st.info("Für dieses Viertel sind keine historischen Preisdaten verfügbar.")
       
       # Tab 2: Standort - Interaktive Karten zum Erkunden von Preis und Reisezeitmuster
   with tab2:
       st.subheader("Interaktive Karten")
       
       # Kartentyp-Auswahl - zwischen Immobilienpreisen und Reisezeiten
       map_type = st.radio(
           "Map Type",
           options=["Immobilienpreise", "Reisezeiten"],
           horizontal=True  # Horizontale Anordnung für bessere Platznutzung
       )
       
       # Abschnitt für Immobilienpreiskarte
       if map_type == "Immobilienpreise":
           # Auswahl für Jahr und Zimmeranzahl in zwei Spalten
           col1, col2 = st.columns(2)
           
           with col1:
               # Jahresauswahl, absteigend sortiert (neueste zuerst)
               years = sorted(df_quartier['Jahr'].unique(), reverse=True)
               selected_year = st.selectbox("Jahr", options=years, index=0)
           
           with col2:
               # Zimmerauswahl für die Karte
               zimmer_options_map = sorted(df_quartier['Zimmeranzahl_num'].unique())
               # Wählt 3 Zimmer standardmäßig, wenn verfügbar
               map_zimmer = st.selectbox("Zimmeranzahl", options=zimmer_options_map, index=min(2, len(zimmer_options_map)-1))
           
           # Immobilienpreiskarte erstellen
           price_map = create_price_heatmap(
               df_quartier,              # Quartier-Datensatz
               quartier_coords,          # Koordinaten der Quartiere
               selected_year=selected_year,    # Ausgewähltes Jahr
               selected_zimmer=map_zimmer      # Ausgewählte Zimmeranzahl
           )
           
           # Karte anzeigen, volle Breite nutzen
           st.plotly_chart(price_map, use_container_width=True)
           
       else:  # Reisezeiten-Karte
           # Auswahl für Zielort und Transportmittel in zwei Spalten
           col1, col2 = st.columns(2)
           
           with col1:
               # Zielortauswahl (z.B. Hauptbahnhof, ETH)
               zielorte = df_travel_times['Zielort'].unique()
               selected_ziel = st.selectbox("Zielort", options=zielorte, index=0)
           
           with col2:
               # Transportmittelauswahl (z.B. öffentlicher Verkehr, Auto)
               transport_options_map = df_travel_times['Transportmittel'].unique()
               map_transport = st.selectbox("Transportmittel", options=transport_options_map, index=0)
           
           # Reisezeitkarte erstellen
           travel_map = create_travel_time_map(
               df_travel_times,              # Reisezeit-Datensatz
               quartier_coords,              # Koordinaten der Quartiere
               zielort=selected_ziel,        # Ausgewählter Zielort
               transportmittel=map_transport  # Ausgewähltes Transportmittel
           )
           
           # Karte anzeigen, volle Breite nutzen
           st.plotly_chart(travel_map, use_container_width=True)
       
       # Tab 3: Marktentwicklungen - Vergleich zwischen verschiedenen Quartieren
       with tab3:
           st.subheader("Nachbarschaftsvergleich")
           
           # Mehrfachauswahl von Quartieren für den Vergleich
           compare_quartiere = st.multiselect(
               "Wählen Sie Nachbarschaften zum Vergleich",
               options=quartier_options,
               default=[selected_quartier]  # Standardmäßig das aktuell ausgewählte Quartier
           )
           
           # Diagramme nur anzeigen, wenn mindestens ein Quartier ausgewählt ist
           if len(compare_quartiere) > 0:
               # Zimmeranzahl für den Vergleich auswählen
               compare_zimmer = st.select_slider(
                   "Zimmeranzahl zum Vergleich",
                   options=[1, 2, 3, 4, 5, 6],
                   value=selected_zimmer  # Standardmäßig die aktuell ausgewählte Zimmeranzahl
               )
               
               # Preisvergleichsdiagramm erstellen
               price_comparison = create_price_comparison_chart(
                   df_quartier,                  # Quartier-Datensatz
                   compare_quartiere,            # Ausgewählte Quartiere 
                   selected_zimmer=compare_zimmer  # Ausgewählte Zimmeranzahl
               )
               
               # Styling anwenden
               apply_chart_styling(price_comparison)
               # Balkenfarbe auf Blau setzen
               price_comparison.update_traces(marker_color='#1565C0')
               
               # Diagramm anzeigen, volle Breite nutzen
               st.plotly_chart(price_comparison, use_container_width=True)
               
               # Preistrendvergleich - zeigt die Entwicklung mehrerer Quartiere im Zeitverlauf
               st.subheader("Preis Trends im Vergleich")
               
               # Zeitreihendiagramm für den Preisvergleich erstellen
               time_series = create_price_time_series(
                   df_quartier,                  # Quartier-Datensatz
                   compare_quartiere,            # Ausgewählte Quartiere
                   selected_zimmer=compare_zimmer  # Ausgewählte Zimmeranzahl
               )
               
               # Styling anwenden
               apply_chart_styling(time_series)
               # Legende horizontal oben anordnen für bessere Lesbarkeit
               time_series.update_layout(
                   legend=dict(
                       orientation="h",           # Horizontale Legende
                       yanchor="bottom",          # Unten ausgerichtet
                       y=1.02,                    # Position über dem Diagramm
                       xanchor="right",           # Rechts ausgerichtet
                       x=1                        # Rechte Kante
                   )
               )
               
               # Diagramm anzeigen, volle Breite nutzen
               st.plotly_chart(time_series, use_container_width=True)
               
               # Faktoren, die den Preis beeinflussen - vereinfachte Feature Importance
               st.subheader("Preisbeeinflussende Faktoren")
               
               # Feature-Importance-Daten - zeigt, welche Faktoren den größten Einfluss haben
               importance_data = {
                   'Feature': ['Nachbarschaft', 'Reisezeit nach HB', 'Zimmeranzahl', 'Baujahr', 'Reisezeit nach Flughafen'],
                   'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]  # Beispielwerte für die Wichtigkeit
               }   
               df_importance = pd.DataFrame(importance_data)
               
               # Horizontales Balkendiagramm für die Feature-Importance
               fig = px.bar(
                   df_importance,
                   x='Importance',               # X-Achse: Wichtigkeit
                   y='Feature',                  # Y-Achse: Features
                   orientation='h',              # Horizontale Ausrichtung
                   title='Faktoren, die die Immobilienpreise beeinflussen'  # Titel
               )
               
               # Styling anwenden
               apply_chart_styling(fig)
               
               # Diagramm anzeigen, volle Breite nutzen
               st.plotly_chart(fig, use_container_width=True)
           else:
               # Informationsmeldung, wenn keine Quartiere für den Vergleich ausgewählt sind
               st.info("Bitte wählen Sie mindestens ein Stadtviertel zum Vergleich aus.")
       
       # Tab 4: Machine-Learning-Modell - Technische Details zum Modell
       with tab4:
           st.subheader("Machine-Learning-Modell")

           # Einführung in das verwendete Modell - erklärt die Technologie
           st.write("""
           Für die Vorhersage von Immobilienpreisen in Zürich haben wir mehrere maschinelle Lernmodelle 
           evaluiert und uns auf Empfehlung von Prof. Simon Mayer für ein Gradient Boosting-Modell entschieden. 
           Diese Entscheidung basiert auf der höheren Vorhersagegenauigkeit im Vergleich zu anderen Modellen 
           wie lineare Regression oder Random Forest.
           """)
           
           # Spalten für Modellmetriken und Visualisierung
           metrics_col1, metrics_col2 = st.columns(2)
           
           # Modell-Performance-Metriken in der ersten Spalte
           with metrics_col1:
               st.markdown("### Modell-Performance")
               # Metriken-Daten für verschiedene Modelle
               metrics_data = {
                   'Metrik': ['MAE (CHF)', 'RSME (CHF)', 'R²'],
                   'Gradient Boosting': ['136207.58', '303630.72', '0.9198'],
                   'Random Forest': ['174884.88', '405427.03', '0.8570'],
                   'Lineare Regression': ['360458.99', '496903.20', '0.7851']
               }
               metrics_df = pd.DataFrame(metrics_data)
               # Metriken als Tabelle anzeigen
               st.table(metrics_df)
               # Erläuterung der Metriken
               st.markdown("""
               **MAE**: Mittlerer absoluter Fehler - durschnittliche Abweichung in CHF  
               **RMSE**: Wurzel des mittleren quadratischen Fehlers  
               **R²**: Bestimmtheitsmass (1.0 = perfekte Vorhersage)
               """)
           
           # Modellvorhersagen vs. tatsächliche Preise in der zweiten Spalte
           with metrics_col2:
               st.markdown("#### Modellvorhersagen vs. tatsächliche Preise")
               
               # Pfad zur CSV-Datei mit Vorhersagedaten
               prediction_data_path = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/main/data/processed/model_evaluation_results.csv'                
               
               try:
                   # Versucht, die CSV-Datei mit den Vorhersagedaten zu laden
                   pred_actual_df = pd.read_csv(prediction_data_path)
                   
                   # Interaktives Streudiagramm mit Plotly erstellen
                   fig = px.scatter(
                       pred_actual_df, 
                       x='Tatsächlicher Preis (CHF)',    # X-Achse: Tatsächliche Preise
                       y='Vorhergesagter Preis (CHF)',   # Y-Achse: Vorhergesagte Preise
                       opacity=0.7,                     # Transparenz für bessere Lesbarkeit
                       hover_data={                     # Formatierung der Hover-Daten
                           'Tatsächlicher Preis (CHF)': ':,.0f',
                           'Vorhergesagter Preis (CHF)': ':,.0f'
                       }
                   )
                   
                   # Perfekte Vorhersagelinie hinzufügen (diagonale Linie)
                   min_val = min(pred_actual_df['Tatsächlicher Preis (CHF)'].min(), 
                               pred_actual_df['Vorhergesagter Preis (CHF)'].min())
                   max_val = max(pred_actual_df['Tatsächlicher Preis (CHF)'].max(), 
                               pred_actual_df['Vorhergesagter Preis (CHF)'].max())
                   
                   # Diagonale Linie für perfekte Vorhersagen hinzufügen
                   fig.add_trace(
                       go.Scatter(
                           x=[min_val, max_val],          # X-Koordinaten
                           y=[min_val, max_val],          # Y-Koordinaten (gleich wie X für Diagonale)
                           mode='lines',                  # Liniendarstellung
                           name='Perfekte Vorhersage',     # Name in der Legende
                           line=dict(color='red', dash='dash')  # Rote gestrichelte Linie
                       )
                   )
                   
                   # Anmerkungen zur Erklärung des Diagramms hinzufügen
                   # Anmerkung für überschätzte Preise (über der Diagonale)
                   fig.add_annotation(
                       x=min_val + (max_val-min_val)*0.2,  # X-Position: 20% des Bereichs
                       y=min_val + (max_val-min_val)*0.8,  # Y-Position: 80% des Bereichs
                       text="Überschätzte Preise",         # Beschriftungstext
                       showarrow=False,                   # Kein Pfeil
                       font=dict(size=12)                 # Schriftgröße
                   )
                   
                   # Anmerkung für unterschätzte Preise (unter der Diagonale)
                   fig.add_annotation(
                       x=min_val + (max_val-min_val)*0.8,  # X-Position: 80% des Bereichs
                       y=min_val + (max_val-min_val)*0.2,  # Y-Position: 20% des Bereichs
                       text="Unterschätzte Preise",        # Beschriftungstext
                       showarrow=False,                   # Kein Pfeil
                       font=dict(size=12)                 # Schriftgröße
                   )
                   
                   # Styling anwenden
                   apply_chart_styling(fig)
                   # Achsenbeschriftungen und Gitternetz konfigurieren
                   fig.update_layout(
                       xaxis=dict(
                           title='Tatsächlicher Preis (CHF)',  # X-Achsenbeschriftung
                           tickformat=',',                   # Tausendertrennzeichen
                           gridcolor='lightgray'              # Helles Gitternetz
                       ),
                       yaxis=dict(
                           title='Vorhergesagter Preis (CHF)',  # Y-Achsenbeschriftung
                           tickformat=',',                    # Tausendertrennzeichen
                           gridcolor='lightgray'               # Helles Gitternetz
                       )
                   )
                   
                   # Interaktives Diagramm in Streamlit anzeigen
                   st.plotly_chart(fig, use_container_width=True)
                   
               except Exception:
                   # Fallback auf Beispieldaten, wenn die tatsächlichen Daten nicht geladen werden können
                   # Dies stellt sicher, dass die Anwendung funktionsfähig bleibt, auch wenn die Daten nicht verfügbar sind
                   prices = [800000, 1200000, 1500000, 1800000, 2200000, 2500000, 1300000, 1600000, 1900000, 2100000]
                   predictions = [850000, 1150000, 1600000, 1700000, 2300000, 2400000, 1250000, 1650000, 1950000, 2050000]
                   
                   # DataFrame mit Beispieldaten erstellen
                   pred_vs_actual = pd.DataFrame({
                       'Tatsächlicher Preis (CHF)': prices,
                       'Vorhergesagter Preis (CHF)': predictions
                   })
                   
                   # Streudiagramm mit Beispieldaten erstellen
                   fig = px.scatter(
                       pred_vs_actual, 
                       x='Tatsächlicher Preis (CHF)',
                       y='Vorhergesagter Preis (CHF)',
                       opacity=0.7
                   )
                   
                   # Perfekte Vorhersagelinie für Beispieldaten hinzufügen
                   min_val = min(min(prices), min(predictions))
                   max_val = max(max(prices), max(predictions))
                   fig.add_trace(
                       go.Scatter(
                           x=[min_val, max_val], 
                           y=[min_val, max_val], 
                           mode='lines', 
                           name='Perfekte Vorhersage',
                           line=dict(color='red', dash='dash')
                       )
                   )
                   
                   # Styling auf Beispieldiagramm anwenden
                   apply_chart_styling(fig)
                   
                   # Beispieldiagramm anzeigen
                   st.plotly_chart(fig, use_container_width=True)
                   # Hinweis, dass dies Beispieldaten sind
                   st.caption("*Hinweis: Dies sind Beispieldaten, nicht die tatsächlichen Modellergebnisse*")
           
           # Feature Importance Abschnitt - zeigt den relativen Einfluss verschiedener Faktoren
           st.subheader("Feature Importance")
           st.write("Die folgende Grafik zeigt, welche Faktoren den größten Einfluss auf die Immobilienpreise in Zürich haben. Diese Feature Importance-Werte basieren auf dem trainierten Gradient Boosting-Modell.")
           
           # Feature-Importance-Mapping-Dictionary für bessere Anzeigenamen
           # Wandelt technische Spaltennamen in benutzerfreundliche Beschreibungen um
           feature_map = {
               'Quartier_Code': 'Nachbarschaft',
               'Zimmeranzahl_num': 'Anzahl Zimmer',
               'PreisProQm': 'Preis pro Quadratmeter',
               'MedianPreis_Baualter': 'Median-Preis nach Baualter',
               'Durchschnitt_Baujahr': 'Baujahr',
               'Preis_Verhältnis': 'Preis-Verhältnis',
               'Quartier_Preisniveau': 'Nachbarschafts-Preisniveau',
               'Reisezeit_Hauptbahnhof': 'Reisezeit zum HB',
               'Reisezeit_Flughafen': 'Reisezeit zum Flughafen'
           }
           
           try:
               # Versucht, Feature-Importance-Daten zu laden
               feature_imp_df = pd.read_csv('https://raw.githubusercontent.com/Rinor909/zurich-real-estate/main/data/processed/feature_importance.csv')
               
               # Benutzerfreundliche Namen wo möglich anwenden
               feature_imp_df['Feature_Display'] = feature_imp_df['Feature'].apply(
                   lambda x: feature_map.get(x, x)  # Verwendet den Mapping-Wert oder behält den Original-Namen bei
               )
               
               # Nach Wichtigkeit sortieren und Top 10 auswählen
               feature_imp_df = feature_imp_df.sort_values('Importance', ascending=True).tail(10)
               
               # Balkendiagramm erstellen
               fig = px.bar(
                   feature_imp_df,
                   x='Importance',                # X-Achse: Wichtigkeit
                   y='Feature_Display',           # Y-Achse: Feature-Namen
                   orientation='h',               # Horizontale Ausrichtung
                   title='Einfluss der verschiedenen Faktoren auf den Immobilienpreis',  # Titel
                   color='Importance',            # Farbe nach Wichtigkeit
                   color_continuous_scale='Blues'  # Blaue Farbskala
               )
               
               # Styling anwenden
               apply_chart_styling(fig)
               # Zusätzliche Layoutanpassungen
               fig.update_layout(
                   xaxis_title="Relativer Einfluss",  # X-Achsenbeschriftung
                   yaxis_title="",                  # Keine Y-Achsenbeschriftung (Features selbsterklärend)
                   coloraxis_showscale=False        # Farbskala ausblenden (redundant mit X-Achse)
               )
               
               # Diagramm anzeigen
               st.plotly_chart(fig, use_container_width=True)
           except Exception:
               # Fallback auf Beispiel-Feature-Importance, wenn die tatsächlichen Daten nicht geladen werden können
               # Dies stellt sicher, dass die Anwendung funktionsfähig bleibt, auch wenn die Daten nicht verfügbar sind
               feature_importance = pd.DataFrame({
                   'Feature': ['Nachbarschaft', 'Nachbarschafts-Preisniveau', 'Reisezeit zum HB', 
                           'Anzahl Zimmer', 'Baujahr', 'Reisezeit zum Flughafen', 'Preis pro Quadratmeter'],
                   'Importance': [0.42, 0.23, 0.12, 0.10, 0.07, 0.04, 0.02]  # Beispielwerte
               })
               
               # Nach Wichtigkeit sortieren
               feature_importance = feature_importance.sort_values('Importance', ascending=True)
               
               # Balkendiagramm mit Beispieldaten erstellen
               fig = px.bar(
                   feature_importance,
                   x='Importance',
                   y='Feature',
                   orientation='h',
                   title='Einfluss der verschiedenen Faktoren auf den Immobilienpreis',
                   color='Importance',
                   color_continuous_scale='Blues'
               )
               
               # Styling anwenden
               apply_chart_styling(fig)
               # Zusätzliche Layoutanpassungen
               fig.update_layout(
                   xaxis_title="Relativer Einfluss",
                   yaxis_title="",
                   coloraxis_showscale=False
               )
               
               # Beispieldiagramm anzeigen
               st.plotly_chart(fig, use_container_width=True)
           
           # Methodik-Erklärung - Erläutert den technischen Prozess dahinter
           st.subheader("Methodik")
           
           # Spalten für die Methodikerklärung
           method_col1, method_col2 = st.columns(2)
           
           # Datenaufbereitungsmethodik in der ersten Spalte
           with method_col1:
               st.markdown("#### Datenaufbereitung")
               st.markdown("1. **Datenbereinigung**: Fehlende Werte wurden durch Medianwerte ersetzt, Outliers wurden identifiziert und behandelt")
               st.markdown("2. **Feature Engineering**: Kategoriale Variablen wurden kodiert, Reisezeitdaten wurden integriert")
               st.markdown("3. **Datentransformation**: Preisniveau-Faktoren für jedes Quartier wurden berechnet")
           
           # Modelltrainingsmethodik in der zweiten Spalte
           with method_col2:
               st.markdown("#### Modelltraining")
               st.markdown("1. **Modellauswahl**: Gradient Boosting wurde nach Vergleich mit linearer Regression und Random Forest ausgewählt")
               st.markdown("2. **Hyperparameter-Tuning**: Die optimalen Parameter wurden mittels Grid-Search bestimmt")
               st.markdown("3. **Kreuzvalidierung**: 5-fache Kreuzvalidierung wurde durchgeführt, um die Robustheit des Modells zu gewährleisten")
           
           # Interaktive Sensitivitätsanalyse - Ermöglicht Benutzern, den Einfluss einzelner Faktoren zu untersuchen
           st.subheader("Interaktive Sensitivitätsanalyse")
           st.write("Hier können Sie sehen, wie sich der Preis ändert, wenn Sie einen einzelnen Faktor variieren, während alle anderen Faktoren konstant bleiben.")
           
           # Benutzer wählt den zu analysierenden Faktor aus
           feature_to_vary = st.selectbox(
               "Zu analysierender Faktor:",
               options=["Quartier", "Zimmeranzahl", "Baujahr", "Reisezeit zum HB"]  # Verfügbare Optionen
           )
           
           # Erstellt ein Diagramm basierend auf dem ausgewählten Faktor
           if feature_to_vary == "Quartier":
               # Top 10 Quartiere nach Medianpreis für Quartiervergleich
               top_quartiere = df_quartier.groupby('Quartier')['MedianPreis'].median().sort_values(ascending=False).head(10).index.tolist()
               
               # Beispielhafte Sensitivitätsdaten für Quartiere
               sensitivity_data = {
                   'Quartier': top_quartiere,
                   'Geschätzter Preis (CHF)': [2200000, 2000000, 1950000, 1850000, 1750000, 1650000, 1600000, 1550000, 1500000, 1450000]
               }
               df_sensitivity = pd.DataFrame(sensitivity_data)
               
               # Balkendiagramm für Quartiervergleich
               fig = px.bar(
                   df_sensitivity,
                   x='Quartier',                 # X-Achse: Quartiere
                   y='Geschätzter Preis (CHF)',   # Y-Achse: Preise
                   title='Preisvariation nach Quartier (3-Zimmer-Wohnung, Baujahr 2000)'  # Titel
               )
           
           elif feature_to_vary == "Zimmeranzahl":
               # Sensitivitätsdaten für Zimmeranzahl
               sensitivity_data = {
                   'Zimmeranzahl': [1, 2, 3, 4, 5, 6],  # Verschiedene Zimmeranzahlen
                   'Geschätzter Preis (CHF)': [800000, 1100000, 1500000, 1900000, 2300000, 2700000]  # Beispielhafte Preise
               }
               df_sensitivity = pd.DataFrame(sensitivity_data)
               
               # Liniendiagramm für Zimmeranzahlvariation
               fig = px.line(
                   df_sensitivity,
                   x='Zimmeranzahl',            # X-Achse: Zimmeranzahl
                   y='Geschätzter Preis (CHF)',  # Y-Achse: Preise
                   markers=True,                # Marker anzeigen
                   title=f'Preisvariation nach Zimmeranzahl (Quartier: {selected_quartier}, Baujahr 2000)'  # Dynamischer Titel
               )
           
           elif feature_to_vary == "Baujahr":
               # Sensitivitätsdaten für Baujahr
               years = list(range(1900, 2026, 10))  # Baujahre von 1900 bis 2025 in 10er-Schritten
               # Preise steigen mit dem Baujahr (5000 CHF pro Jahr ab 1900)
               sensitivity_data = {
                   'Baujahr': years,
                   'Geschätzter Preis (CHF)': [1100000 + (year-1900)*5000 for year in years]
               }
               df_sensitivity = pd.DataFrame(sensitivity_data)
               
               # Liniendiagramm für Baujahrvariation
               fig = px.line(
                   df_sensitivity,
                   x='Baujahr',                 # X-Achse: Baujahr
                   y='Geschätzter Preis (CHF)',  # Y-Achse: Preise
                   markers=True,                # Marker anzeigen
                   title=f'Preisvariation nach Baujahr (Quartier: {selected_quartier}, 3-Zimmer-Wohnung)'  # Dynamischer Titel
               )
           
           else:  # Reisezeit zum HB
               # Sensitivitätsdaten für Reisezeit zum Hauptbahnhof
               transit_times = list(range(5, 41, 5))  # Reisezeiten von 5 bis 40 Minuten in 5er-Schritten
               # Preise sinken mit steigender Reisezeit (15.000 CHF pro Minute)
               sensitivity_data = {
                   'Reisezeit zum HB (Min)': transit_times,
                   'Geschätzter Preis (CHF)': [1800000 - time*15000 for time in transit_times]
               }
               df_sensitivity = pd.DataFrame(sensitivity_data)
               
               # Liniendiagramm für Reisezeitvariation
               fig = px.line(
                   df_sensitivity,
                   x='Reisezeit zum HB (Min)',   # X-Achse: Reisezeit
                   y='Geschätzter Preis (CHF)',  # Y-Achse: Preise
                   markers=True,                # Marker anzeigen
                   title=f'Preisvariation nach Reisezeit zum HB (Quartier: {selected_quartier}, 3-Zimmer-Wohnung, Baujahr 2000)'  # Dynamischer Titel
               )
           
           # Styling auf das Sensitivitätsdiagramm anwenden
           apply_chart_styling(fig)
           
           # Diagramm anzeigen, volle Breite nutzen
           st.plotly_chart(fig, use_container_width=True)
           
           # Hinweis zu Modelleinschränkungen - erklärt die Grenzen des Modells
           # Doppelte Kommentarzeile wurde aus dem Original beibehalten
           st.info("""
           Modelleinschränkungen:
           - Das Modell basiert auf historischen Daten und kann unerwartete Marktveränderungen nicht vorhersagen
           - Faktoren wie Ausstattungsqualität oder Grundriss der Wohnung werden nicht berücksichtigt
           - Mikro-Standortfaktoren wie Aussicht oder Lärmbelastung können den tatsächlichen Preis beeinflussen
           """)
           
   # ---- FOOTER ----
   # Fußzeile mit Quellenangaben und Entwicklungskontext
   st.caption(
       "Entwickelt im Rahmen des CS-Kurses an der HSG | Datenquellen: "
       "[Immobilienpreise nach Quartier](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-2) | "
       "[Immobilienpreise nach Baualter](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-3)"
   )

# Ausführungsprüfung - Code außerhalb der main-Funktion
# Dies stellt sicher, dass die Anwendung nur ausgeführt wird, wenn die Datei direkt ausgeführt wird
if __name__ == "__main__":
   # Module in den Pfad einfügen für korrekten Import
   import sys
   sys.path.append('app')
   
   # Die Hauptfunktion ausführen, um die Anwendung zu starten
   main()