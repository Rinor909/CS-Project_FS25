import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_price_heatmap(df_quartier, quartier_coords, selected_year=2024, selected_zimmer=3):
    """
    Erstellt eine Heatmap der Immobilienpreise in Zürich
    
    Args:
        df_quartier (pd.DataFrame): DataFrame mit Quartierdaten
        quartier_coords (dict): Dictionary mit Koordinaten der Quartiere
        selected_year (int): Ausgewähltes Jahr
        selected_zimmer (int): Ausgewählte Zimmeranzahl
        
    Returns:
        plotly.graph_objects.Figure: Plotly-Figur mit der Heatmap
    """
    # Daten für das ausgewählte Jahr und die ausgewählte Zimmeranzahl filtern
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == selected_year) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer)
    ].copy()
    
    # Sicherstellen, dass jedes Quartier nur einmal vorkommt (Durchschnitt, falls mehrere Einträge)
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Koordinaten für jedes Quartier hinzufügen
    df_map = pd.DataFrame(columns=['Quartier', 'MedianPreis', 'PreisProQm', 'lat', 'lon'])
    
    for _, row in df_grouped.iterrows():
        quartier = row['Quartier']
        if quartier in quartier_coords:
            coords = quartier_coords[quartier]
            df_map = pd.concat([df_map, pd.DataFrame({
                'Quartier': [quartier],
                'MedianPreis': [row['MedianPreis']],
                'PreisProQm': [row['PreisProQm']],
                'lat': [coords['lat']],
                'lon': [coords['lng']]
            })], ignore_index=True)
    
    # Farskala für die Preise festlegen
    min_price = df_map['MedianPreis'].min()
    max_price = df_map['MedianPreis'].max()
    
    # Plot erstellen
    fig = px.scatter_mapbox(
        df_map, 
        lat='lat', 
        lon='lon',
        color='MedianPreis',
        size='MedianPreis',
        size_max=20,
        hover_name='Quartier',
        hover_data={
            'MedianPreis': True,
            'PreisProQm': True,
            'lat': False,
            'lon': False
        },
        color_continuous_scale='Viridis',
        range_color=[min_price, max_price],
        zoom=11,
        height=600,
        width=800,
        title=f'Immobilienpreise in Zürich ({selected_year}, {selected_zimmer} Zimmer)'
    )
    
    # Layout anpassen
    fig.update_layout(
        mapbox_style='open-street-map',
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(
            title='Preis (CHF)',
            tickformat=',.0f'
        )
    )
    
    return fig

def create_travel_time_map(df_travel_times, quartier_coords, zielort='Hauptbahnhof', transportmittel='transit'):
    """
    Erstellt eine Karte mit den Reisezeiten zu einem bestimmten Zielort
    
    Args:
        df_travel_times (pd.DataFrame): DataFrame mit Reisezeitdaten
        quartier_coords (dict): Dictionary mit Koordinaten der Quartiere
        zielort (str): Ausgewählter Zielort
        transportmittel (str): Ausgewähltes Transportmittel
        
    Returns:
        plotly.graph_objects.Figure: Plotly-Figur mit der Reisezeit-Karte
    """
    # Daten für den ausgewählten Zielort und das ausgewählte Transportmittel filtern
    df_filtered = df_travel_times[
        (df_travel_times['Zielort'] == zielort) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ].copy()
    
    # Koordinaten für jedes Quartier hinzufügen
    df_map = pd.DataFrame(columns=['Quartier', 'Reisezeit_Minuten', 'lat', 'lon'])
    
    for _, row in df_filtered.iterrows():
        quartier = row['Quartier']
        if quartier in quartier_coords:
            coords = quartier_coords[quartier]
            df_map = pd.concat([df_map, pd.DataFrame({
                'Quartier': [quartier],
                'Reisezeit_Minuten': [row['Reisezeit_Minuten']],
                'lat': [coords['lat']],
                'lon': [coords['lng']]
            })], ignore_index=True)
    
    # Plot erstellen
    fig = px.scatter_mapbox(
        df_map, 
        lat='lat', 
        lon='lon',
        color='Reisezeit_Minuten',
        size='Reisezeit_Minuten',
        size_max=20,
        hover_name='Quartier',
        hover_data={
            'Reisezeit_Minuten': True,
            'lat': False,
            'lon': False
        },
        color_continuous_scale='Cividis_r',  # Umgekehrte Farbskala (dunkel = lange Reisezeit)
        zoom=11,
        height=600,
        width=800,
        title=f'Reisezeit nach {zielort} ({transportmittel})'
    )
    
    # Layout anpassen
    fig.update_layout(
        mapbox_style='open-street-map',
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(
            title='Minuten',
            tickformat=',.0f'
        )
    )
    
    return fig

def create_price_comparison_chart(df_quartier, selected_quartiere, selected_zimmer=3):
    """
    Erstellt ein Balkendiagramm zum Vergleich der Preise in verschiedenen Quartieren
    
    Args:
        df_quartier (pd.DataFrame): DataFrame mit Quartierdaten
        selected_quartiere (list): Liste der ausgewählten Quartiere
        selected_zimmer (int): Ausgewählte Zimmeranzahl
        
    Returns:
        plotly.graph_objects.Figure: Plotly-Figur mit dem Balkendiagramm
    """
    # Neueste Daten für die ausgewählten Quartiere und Zimmeranzahl filtern
    neuestes_jahr = df_quartier['Jahr'].max()
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == neuestes_jahr) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # Durchschnittliche Preise pro Quartier berechnen
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Quartiere nach Preis sortieren
    df_grouped = df_grouped.sort_values('MedianPreis', ascending=False)
    
    # Zwei Balkendiagramme erstellen: MedianPreis und PreisProQm
    fig = go.Figure()
    
    # MedianPreis-Balken
    fig.add_trace(go.Bar(
        x=df_grouped['Quartier'],
        y=df_grouped['MedianPreis'],
        name='Median-Kaufpreis (CHF)',
        marker_color='royalblue',
        text=df_grouped['MedianPreis'].apply(lambda x: f'{x:,.0f} CHF'),
        textposition='auto'
    ))
    
    # Layout anpassen
    fig.update_layout(
        title=f'Immobilienpreisvergleich ({neuestes_jahr}, {selected_zimmer} Zimmer)',
        xaxis_title='Quartier',
        yaxis_title='Preis (CHF)',
        height=400,
        width=800,
        barmode='group',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_price_time_series(df_quartier, selected_quartiere, selected_zimmer=3):
    """
    Erstellt ein Liniendiagramm zur Preisentwicklung über die Zeit
    
    Args:
        df_quartier (pd.DataFrame): DataFrame mit Quartierdaten
        selected_quartiere (list): Liste der ausgewählten Quartiere
        selected_zimmer (int): Ausgewählte Zimmeranzahl
        
    Returns:
        plotly.graph_objects.Figure: Plotly-Figur mit dem Liniendiagramm
    """
    # Daten für die ausgewählten Quartiere und Zimmeranzahl filtern
    df_filtered = df_quartier[
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # Nach Jahr und Quartier gruppieren
    df_grouped = df_filtered.groupby(['Jahr', 'Quartier']).agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Plot erstellen
    fig = px.line(
        df_grouped, 
        x='Jahr', 
        y='MedianPreis', 
        color='Quartier',
        markers=True,
        title=f'Preisentwicklung ({selected_zimmer} Zimmer)',
        height=400,
        width=800
    )
    
    # Layout anpassen
    fig.update_layout(
        xaxis_title='Jahr',
        yaxis_title='Median-Kaufpreis (CHF)',
        legend_title='Quartier',
        yaxis=dict(tickformat=',.0f'),
        hovermode='x unified'
    )
    
    return fig