import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_price_heatmap(df_quartier, quartier_coords, selected_year=2024, selected_zimmer=3):
    """
    Creates a heatmap of real estate prices in Zurich
    
    Args:
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        quartier_coords (dict): Dictionary with neighborhood coordinates
        selected_year (int): Selected year
        selected_zimmer (int): Selected number of rooms
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the heatmap
    """
    # Filter data for the selected year and number of rooms
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == selected_year) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer)
    ].copy()
    
    # Ensure each neighborhood only appears once (average if multiple entries)
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Add coordinates for each neighborhood
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
    
    # Set color scale for prices
    min_price = df_map['MedianPreis'].min()
    max_price = df_map['MedianPreis'].max()
    
    # Create plot
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
        title=f'Real Estate Prices in Zurich ({selected_year}, {selected_zimmer} rooms)'
    )
    
    # Adjust layout
    fig.update_layout(
        mapbox_style='open-street-map',
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(
            title='Price (CHF)',
            tickformat=',.0f'
        )
    )
    
    return fig

def create_travel_time_map(df_travel_times, quartier_coords, zielort='Hauptbahnhof', transportmittel='transit'):
    """
    Creates a map with travel times to a specific destination
    
    Args:
        df_travel_times (pd.DataFrame): DataFrame with travel time data
        quartier_coords (dict): Dictionary with neighborhood coordinates
        zielort (str): Selected destination
        transportmittel (str): Selected transport mode
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the travel time map
    """
    # Filter data for the selected destination and transport mode
    df_filtered = df_travel_times[
        (df_travel_times['Zielort'] == zielort) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ].copy()
    
    # Add coordinates for each neighborhood
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
        color_continuous_scale='Cividis_r',  # Reversed color scale (dark = long travel time)
        zoom=11,
        height=600,
        width=800,
        title=f'Travel Time to {zielort} ({transportmittel})'
    )
    
    # Adjust layout
    fig.update_layout(
        mapbox_style='open-street-map',
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar=dict(
            title='Minutes',
            tickformat=',.0f'
        )
    )
    
    return fig

def create_price_comparison_chart(df_quartier, selected_quartiere, selected_zimmer=3):
    """
    Creates a bar chart to compare prices across different neighborhoods
    
    Args:
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        selected_quartiere (list): List of selected neighborhoods
        selected_zimmer (int): Selected number of rooms
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the bar chart
    """
    # Filter latest data for the selected neighborhoods and number of rooms
    neuestes_jahr = df_quartier['Jahr'].max()
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == neuestes_jahr) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # If no data available, return empty figure
    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available for the selected parameters",
            height=400,
            width=800
        )
        return fig
    
    # Calculate average prices per neighborhood
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Sort neighborhoods by price
    df_grouped = df_grouped.sort_values('MedianPreis', ascending=False)
    
    # Create two bar charts: MedianPreis and PreisProQm
    fig = go.Figure()
    
    # MedianPreis bars
    fig.add_trace(go.Bar(
        x=df_grouped['Quartier'],
        y=df_grouped['MedianPreis'],
        name='Median Purchase Price (CHF)',
        marker_color='royalblue',
        text=df_grouped['MedianPreis'].apply(lambda x: f'{x:,.0f} CHF'),
        textposition='auto'
    ))
    
    # Adjust layout
    fig.update_layout(
        title=f'Real Estate Price Comparison ({neuestes_jahr}, {selected_zimmer} rooms)',
        xaxis_title='Neighborhood',
        yaxis_title='Price (CHF)',
        height=400,
        width=800,
        barmode='group',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_price_time_series(df_quartier, selected_quartiere, selected_zimmer=3):
    """
    Creates a line chart showing price development over time
    
    Args:
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        selected_quartiere (list): List of selected neighborhoods
        selected_zimmer (int): Selected number of rooms
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the line chart
    """
    # Filter data for the selected neighborhoods and number of rooms
    df_filtered = df_quartier[
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
        (df_quartier['Quartier'].isin(selected_quartiere))
    ].copy()
    
    # If no data available, return empty figure
    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available for the selected parameters",
            height=400,
            width=800
        )
        return fig
    
    # Group by year and neighborhood
    df_grouped = df_filtered.groupby(['Jahr', 'Quartier']).agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean'
    }).reset_index()
    
    # Create plot
    fig = px.line(
        df_grouped, 
        x='Jahr', 
        y='MedianPreis', 
        color='Quartier',
        markers=True,
        title=f'Price Development ({selected_zimmer} rooms)',
        height=400,
        width=800
    )
    
    # Adjust layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Median Purchase Price (CHF)',
        legend_title='Neighborhood',
        yaxis=dict(tickformat=',.0f'),
        hovermode='x unified'
    )
    
    return fig