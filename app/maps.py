# Notwendige Bibliotheken importieren
import pandas as pd               # Für Datenmanipulation und -analyse
import numpy as np                # Für numerische Operationen
import streamlit as st            # Für die Erstellung der Webapplikationsoberfläche
import plotly.express as px       # Für interaktive Visualisierungen mit weniger Code
import plotly.graph_objects as go # Für spezifischere interaktive Visualisierungen

def create_price_heatmap(df_quartier, quartier_coords, selected_year=2024, selected_zimmer=3):
   """Erstellt eine Heatmap der Immobilienpreise in Zürich"""
   # Überprüfung, ob die nötigen Daten vorhanden sind
   # Falls keine Daten verfügbar sind, wird eine leere Grafik mit Fehlermeldung zurückgegeben
   if 'Quartier' not in df_quartier.columns or df_quartier.empty:
       return go.Figure().update_layout(title="Keine Daten verfügbar")
   
   # Filtert die Daten nach dem ausgewählten Jahr und der Zimmeranzahl
   # Eine Kopie wird erstellt, um Warnungen zu vermeiden
   df_filtered = df_quartier[
       (df_quartier['Jahr'] == selected_year) & 
       (df_quartier['Zimmeranzahl_num'] == selected_zimmer)
   ].copy()
   
   # Gruppierung nach Quartier und Berechnung der Durchschnittswerte
   # Dies stellt sicher, dass jedes Quartier nur einmal auf der Karte erscheint
   df_grouped = df_filtered.groupby('Quartier').agg({
       'MedianPreis': 'mean',  # Durchschnittlicher Medianpreis pro Quartier
       'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
   }).reset_index()
   
   # Erstellung eines neuen DataFrames für Kartenvisualisierung mit Koordinaten
   df_map = pd.DataFrame(columns=['Quartier', 'MedianPreis', 'PreisProQm', 'lat', 'lon'])
   
   # Fügt jedes Quartier zum Karten-DataFrame hinzu, wenn Koordinaten vorhanden sind
   for _, row in df_grouped.iterrows():
       quartier = row['Quartier']
       if quartier in quartier_coords:  # Überspringt Quartiere ohne Koordinatendaten
           coords = quartier_coords[quartier]
           # Erstellt ein Dictionary mit Quartierdaten und Koordinaten
           new_row = {
               'Quartier': quartier,
               'MedianPreis': row['MedianPreis'],
               'lat': coords['lat'],   # Breitengrad für die Kartendarstellung
               'lon': coords['lng']    # Längengrad für die Kartendarstellung
           }
           # Fügt Preis pro Quadratmeter hinzu, falls verfügbar
           if 'PreisProQm' in row and not pd.isna(row['PreisProQm']):
               new_row['PreisProQm'] = row['PreisProQm']
           
           # Hängt die neue Zeile an den Karten-DataFrame an
           df_map = pd.concat([df_map, pd.DataFrame([new_row])], ignore_index=True)
   
   # Konfiguriert, welche Daten beim Überfahren der Punkte angezeigt werden
   # Koordinatenwerte werden ausgeblendet, da sie für Benutzer nicht aussagekräftig sind
   hover_data = {'MedianPreis': True, 'lat': False, 'lon': False}
   if 'PreisProQm' in df_map.columns:
       hover_data['PreisProQm'] = True  # Fügt Preis pro m² hinzu, falls verfügbar
       
   # Erstellt die interaktive Streuungskarte mit Plotly Express
   fig = px.scatter_mapbox(
       df_map,                        # Datenquelle
       lat='lat',                     # Breitengrad-Spalte
       lon='lon',                     # Längengrad-Spalte
       color='MedianPreis',           # Färbt Punkte nach Preis
       size='MedianPreis',            # Größe der Punkte nach Preis (höhere Preise = größere Punkte)
       size_max=20,                   # Maximale Punktgröße
       hover_name='Quartier',         # Zeigt Quartiername beim Überfahren
       hover_data=hover_data,         # Konfiguriert zusätzliche Hover-Daten
       color_continuous_scale='Viridis', # Farbskala (lila bis gelb)
       zoom=11,                       # Anfängliche Zoomstufe
       height=600,                    # Höhe der Grafik in Pixeln
       title=f'Immobilienpreise in Zürich ({selected_year}, {selected_zimmer} Zimmer)' # Titel mit dynamischen Werten
   )
   
   # Anpassung des Kartenaussehens und Layouts
   fig.update_layout(
       mapbox_style='open-street-map',  # Verwendung von OpenStreetMap-Kacheln
       margin={"r":0, "t":50, "l":0, "b":0},  # Anpassung der Ränder für bessere Raumnutzung
       coloraxis_colorbar=dict(           # Konfiguration der Farbskalenlegende
           title='Preis (CHF)',           # Titel für die Farbskala
           tickformat=',.0f'              # Formatierung der Zahlen mit Tausendertrennzeichen
       )
   )
   return fig  # Gibt die fertige Visualisierung zurück

def create_travel_time_map(df_travel_times, quartier_coords, zielort='Hauptbahnhof', transportmittel='transit'):
   """Erstellt eine Karte mit Reisezeiten zu einem bestimmten Zielort"""
   # Filtert Reisezeitdaten für den ausgewählten Zielort und Transportmittel
   df_filtered = df_travel_times[
       (df_travel_times['Zielort'] == zielort) & 
       (df_travel_times['Transportmittel'] == transportmittel)
   ].copy()
   
   # Erstellung eines neuen DataFrames für Kartenvisualisierung mit Koordinaten
   df_map = pd.DataFrame(columns=['Quartier', 'Reisezeit_Minuten', 'lat', 'lon'])
   
   # Fügt jedes Quartier zum Karten-DataFrame hinzu, wenn Koordinaten vorhanden sind
   for _, row in df_filtered.iterrows():
       quartier = row['Quartier']
       if quartier in quartier_coords:  # Überspringt Quartiere ohne Koordinatendaten
           coords = quartier_coords[quartier]
           # Fügt Quartier, Reisezeit und Koordinaten zum Karten-DataFrame hinzu
           df_map = pd.concat([df_map, pd.DataFrame({
               'Quartier': [quartier],
               'Reisezeit_Minuten': [row['Reisezeit_Minuten']],
               'lat': [coords['lat']],   # Breitengrad für die Kartendarstellung
               'lon': [coords['lng']]    # Längengrad für die Kartendarstellung
           })], ignore_index=True)
   
   # Erstellt die interaktive Streuungskarte mit Plotly Express
   fig = px.scatter_mapbox(
       df_map,                        # Datenquelle
       lat='lat',                     # Breitengrad-Spalte
       lon='lon',                     # Längengrad-Spalte
       color='Reisezeit_Minuten',     # Färbt Punkte nach Reisezeit
       size='Reisezeit_Minuten',      # Größe der Punkte nach Reisezeit (längere Zeiten = größere Punkte)
       size_max=20,                   # Maximale Punktgröße
       hover_name='Quartier',         # Zeigt Quartiername beim Überfahren
       hover_data={                   # Konfiguriert Hover-Daten
           'Reisezeit_Minuten': True, # Zeigt Reisezeit beim Überfahren
           'lat': False,              # Versteckt Breitengrad beim Überfahren
           'lon': False               # Versteckt Längengrad beim Überfahren
       },
       color_continuous_scale='Cividis_r',  # Umgekehrte Farbskala (dunkel = lange Reisezeit)
       zoom=11,                            # Anfängliche Zoomstufe
       height=600,                         # Höhe der Grafik in Pixeln
       title=f'Reisezeit nach {zielort} ({transportmittel})' # Titel mit dynamischen Werten
   )
   
   # Anpassung des Kartenaussehens und Layouts
   fig.update_layout(
       mapbox_style='open-street-map',  # Verwendung von OpenStreetMap-Kacheln
       margin={"r":0, "t":50, "l":0, "b":0},  # Anpassung der Ränder für bessere Raumnutzung
       coloraxis_colorbar=dict(         # Konfiguration der Farbskalenlegende
           title='Minuten',             # Titel für die Farbskala (Minuten)
           tickformat=',.0f'            # Formatierung der Zahlen mit Tausendertrennzeichen
       )
   )
   
   return fig  # Gibt die fertige Visualisierung zurück

def create_price_comparison_chart(df_quartier, selected_quartiere, selected_zimmer=3):
   """Erstellt ein Balkendiagramm zum Vergleich der Preise in verschiedenen Quartieren"""
   # Findet das aktuellste Jahr im Datensatz für aktuelle Preise
   neuestes_jahr = df_quartier['Jahr'].max()
   
   # Filtert den Datensatz nach:
   # - Immobilien aus dem aktuellsten Jahr
   # - Immobilien mit der ausgewählten Zimmeranzahl
   # - Immobilien in den ausgewählten Quartieren für den Vergleich
   df_filtered = df_quartier[
       (df_quartier['Jahr'] == neuestes_jahr) & 
       (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
       (df_quartier['Quartier'].isin(selected_quartiere))
   ].copy()
   
   # Gruppierung nach Quartier und Berechnung der Durchschnittswerte
   # Dies stellt einheitliche Vergleichswerte für jedes Quartier sicher
   df_grouped = df_filtered.groupby('Quartier').agg({
       'MedianPreis': 'mean',  # Durchschnittlicher Medianpreis pro Quartier
       'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
   }).reset_index()
   
   # Sortiert Quartiere nach Preis von höchstem zu niedrigstem
   # Dies verbessert die Lesbarkeit und Interpretierbarkeit des Diagramms
   df_grouped = df_grouped.sort_values('MedianPreis', ascending=False)
   
   # Erstellt ein leeres Figurobjekt
   fig = go.Figure()
   
   # Fügt einen Balkendiagrammtrace zur Figur hinzu
   fig.add_trace(go.Bar(
       x=df_grouped['Quartier'],      # X-Achse: Quartiere
       y=df_grouped['MedianPreis'],   # Y-Achse: Medianpreise
       name='Median Kaufpreis (CHF)', # Name für die Legende
       marker_color='royalblue',      # Setzt Balkenfarbe
       text=df_grouped['MedianPreis'].apply(lambda x: f'{x:,.0f} CHF'),  # Text auf den Balken
       textposition='auto'            # Automatische Textpositionierung
   ))
   
   # Anpassung des Diagrammaussehens und Layouts
   fig.update_layout(
       title=f'Immobilienpreisvergleich ({neuestes_jahr}, {selected_zimmer} Zimmer)',  # Dynamischer Titel
       xaxis_title='Quartier',        # X-Achsentitel
       yaxis_title='Preis (CHF)',     # Y-Achsentitel
       height=400,                    # Höhe der Grafik in Pixeln
       barmode='group',               # Gruppierungsmodus für mehrere Balken (für zukünftige Erweiterungen)
       xaxis={'categoryorder': 'total descending'}  # Sortiert Quartiere nach Preis
   )
   
   return fig  # Gibt die fertige Visualisierung zurück

def create_price_time_series(df_quartier, selected_quartiere, selected_zimmer=3):
   """Erstellt ein Liniendiagramm zur Darstellung der Preisentwicklung im Zeitverlauf"""
   # Filtert den Datensatz nach:
   # - Immobilien mit der ausgewählten Zimmeranzahl
   # - Immobilien in den ausgewählten Quartieren für den Vergleich
   # Hinweis: Im Gegensatz zum Balkendiagramm behalten wir alle Jahre, um Trends zu zeigen
   df_filtered = df_quartier[
       (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
       (df_quartier['Quartier'].isin(selected_quartiere))
   ].copy()
   
   # Gruppiert Daten nach Jahr und Quartier, berechnet dann die Durchschnittswerte
   # Diese Aggregation erstellt einen Datenpunkt pro Jahr und Quartier
   df_grouped = df_filtered.groupby(['Jahr', 'Quartier']).agg({
       'MedianPreis': 'mean',  # Durchschnittlicher Medianpreis für jedes Jahr/Quartier
       'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
   }).reset_index()
   
   # Erstellt ein Liniendiagramm mit Preistrends über die Zeit mittels Plotly Express
   fig = px.line(
       df_grouped,                    # Datenquelle
       x='Jahr',                      # X-Achse: Jahre
       y='MedianPreis',               # Y-Achse: Medianpreise
       color='Quartier',              # Färbt Linien nach Quartier
       markers=True,                  # Fügt Marker bei jedem Datenpunkt hinzu
       title=f'Preisentwicklung ({selected_zimmer} Zimmer)',  # Dynamischer Titel
       height=400                     # Höhe der Grafik in Pixeln
   )
   
   # Anpassung des Diagrammaussehens und Layouts
   fig.update_layout(
       xaxis_title='Jahr',                # X-Achsentitel
       yaxis_title='Median Kaufpreis (CHF)',  # Y-Achsentitel
       legend_title='Quartier',           # Legendentitel
       yaxis=dict(tickformat=',.0f'),     # Formatiert Y-Achse mit Tausendertrennzeichen
       hovermode='x unified'              # Zeigt alle Werte für dasselbe Jahr beim Überfahren
   )
   
   return fig  # Gibt die fertige Visualisierung zurück