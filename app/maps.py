import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_price_heatmap(df_quartier, quartier_coords, selected_year=2024, selected_zimmer=3):
    """Creates a heatmap of real estate prices in Zurich"""
    # Check if necessary data is available
    if 'Quartier' not in df_quartier.columns or df_quartier.empty:
        return go.Figure().update_layout(title="Keine Daten verfügbar")
    
    # Filter data 
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == selected_year) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer)
    ].copy()
    
    # Group by neighborhood
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
    }).reset_index()
    
    # Add coordinates 
    df_map = pd.DataFrame(columns=['Quartier', 'MedianPreis', 'PreisProQm', 'lat', 'lon'])
    for _, row in df_grouped.iterrows():
        quartier = row['Quartier']
        if quartier in quartier_coords:
            coords = quartier_coords[quartier]
            new_row = {
                'Quartier': quartier,
                'MedianPreis': row['MedianPreis'],
                'lat': coords['lat'],
                'lon': coords['lng']
            }
            if 'PreisProQm' in row and not pd.isna(row['PreisProQm']):
                new_row['PreisProQm'] = row['PreisProQm']
            
            df_map = pd.concat([df_map, pd.DataFrame([new_row])], ignore_index=True)
    
    # Create plot
    hover_data = {'MedianPreis': True, 'lat': False, 'lon': False}
    if 'PreisProQm' in df_map.columns:
        hover_data['PreisProQm'] = True
        
    fig = px.scatter_mapbox(
        df_map, 
        lat='lat', 
        lon='lon',
        color='MedianPreis',
        size='MedianPreis',
        size_max=20,
        hover_name='Quartier',
        hover_data=hover_data,
        color_continuous_scale='Viridis',
        zoom=11,
        height=600,
        title=f'Immobilienpreise in Zürich ({selected_year}, {selected_zimmer} Zimmer)'
    )
    
    # Adjust layout
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
    """Creates a map with travel times to a specific destination"""
    # Filter data
    df_filtered = df_travel_times[
        (df_travel_times['Zielort'] == zielort) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ].copy()
    
    # Add coordinates
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
    
    # Create plot
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
        color_continuous_scale='Cividis_r',  # Reversed color scale
        zoom=11,
        height=600,
        title=f'Reisezeit nach {zielort} ({transportmittel})'
    )
    
    # Adjust layout
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
    """Creates a bar chart to compare prices across different neighborhoods"""
    # Get latest data year
    neuestes_jahr = df_quartier['Jahr'].max()
    
    # Filter data
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == neuestes_jahr) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # Calculate average prices per neighborhood
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
    }).reset_index()
    
    # Sort neighborhoods by price
    df_grouped = df_grouped.sort_values('MedianPreis', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_grouped['Quartier'],
        y=df_grouped['MedianPreis'],
        name='Median Kaufpreis (CHF)',
        marker_color='royalblue',
        text=df_grouped['MedianPreis'].apply(lambda x: f'{x:,.0f} CHF'),
        textposition='auto'
    ))
    
    # Adjust layout
    fig.update_layout(
        title=f'Immobilienpreisvergleich ({neuestes_jahr}, {selected_zimmer} Zimmer)',
        xaxis_title='Quartier',
        yaxis_title='Preis (CHF)',
        height=400,
        barmode='group',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_price_time_series(df_quartier, selected_quartiere, selected_zimmer=3):
    """Creates a line chart showing price development over time"""
    # Filter data
    df_filtered = df_quartier[
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # Group by year and neighborhood
    df_grouped = df_filtered.groupby(['Jahr', 'Quartier']).agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
    }).reset_index()
    
    # Create plot
    fig = px.line(
        df_grouped, 
        x='Jahr', 
        y='MedianPreis', 
        color='Quartier',
        markers=True,
        title=f'Preisentwicklung ({selected_zimmer} Zimmer)',
        height=400
    )
    
    # Adjust layout
    fig.update_layout(
        xaxis_title='Jahr',
        yaxis_title='Median Kaufpreis (CHF)',
        legend_title='Quartier',
        yaxis=dict(tickformat=',.0f'),
        hovermode='x unified'
    )
    
    return fig