import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Define dark theme colors
DARK_BG = "#121212"
DARK_CARD_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
GRID_COLOR = "#333333"
ZURICH_BLUE = "#0038A8"  # Deep blue from Zürich flag

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
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Jahr', 'Zimmeranzahl_num', 'MedianPreis']
    if any(col not in df_quartier.columns for col in required_columns) or df_quartier.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Keine Daten verfügbar. Bitte führen Sie die Datenvorbereitungsskripte aus.",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    # Filter data for the selected year and number of rooms
    df_filtered = df_quartier[
        (df_quartier['Jahr'] == selected_year) & 
        (df_quartier['Zimmeranzahl_num'] == selected_zimmer)
    ].copy()
    
    # If no data for this selection, return empty figure
    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Keine Daten für Jahr {selected_year} und {selected_zimmer} Zimmer verfügbar",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    # Ensure each neighborhood only appears once (average if multiple entries)
    df_grouped = df_filtered.groupby('Quartier').agg({
        'MedianPreis': 'mean',
        'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
    }).reset_index()
    
    # Add coordinates for each neighborhood
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
    
    # If no mappable data, return empty figure
    if df_map.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Keine kartierbaren Daten verfügbar. Überprüfen Sie die Quartierkoordinaten.",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    # Set color scale for prices
    min_price = df_map['MedianPreis'].min()
    max_price = df_map['MedianPreis'].max()
    
    # Create plot
    try:
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
            color_continuous_scale='Blues',  # Use blue for Zurich
            range_color=[min_price, max_price],
            zoom=11,
            height=600,
            width=800,
            title=f'Immobilienpreise in Zürich ({selected_year}, {selected_zimmer} Zimmer)'
        )
        
        # Adjust layout with dark theme
        fig.update_layout(
            mapbox_style='dark',  # Dark map style
            margin={"r":0, "t":50, "l":0, "b":0},
            coloraxis_colorbar=dict(
                title='Preis (CHF)',
                tickformat=',.0f',
                titlefont=dict(color=DARK_TEXT),
                tickfont=dict(color=DARK_TEXT)
            ),
            font=dict(color=DARK_TEXT),
            paper_bgcolor=DARK_CARD_BG
        )
        
        return fig
    except Exception as e:
        print(f"Error creating price heatmap: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler bei der Kartenerstellung: {str(e)}",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
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
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten']
    if any(col not in df_travel_times.columns for col in required_columns) or df_travel_times.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Keine Reisezeitdaten verfügbar. Bitte führen Sie generate_travel_times.py aus.",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    # Filter data for the selected destination and transport mode
    df_filtered = df_travel_times[
        (df_travel_times['Zielort'] == zielort) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ].copy()
    
    # If no data for this selection, return empty figure
    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Keine Reisezeitdaten verfügbar für {zielort} mit {transportmittel}",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
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
    
    # If no mappable data, return empty figure
    if df_map.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Keine kartierbaren Reisezeitdaten verfügbar. Überprüfen Sie die Quartierkoordinaten.",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    # Create plot
    try:
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
            color_continuous_scale='Blues_r',  # Reversed blue color scale
            zoom=11,
            height=600,
            width=800,
            title=f'Reisezeit nach {zielort} ({transportmittel})'
        )
        
        # Adjust layout with dark theme
        fig.update_layout(
            mapbox_style='dark',  # Dark map style
            margin={"r":0, "t":50, "l":0, "b":0},
            coloraxis_colorbar=dict(
                title='Minuten',
                tickformat=',.0f',
                titlefont=dict(color=DARK_TEXT),
                tickfont=dict(color=DARK_TEXT)
            ),
            font=dict(color=DARK_TEXT),
            paper_bgcolor=DARK_CARD_BG
        )
        
        return fig
    except Exception as e:
        print(f"Error creating travel time map: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler bei der Kartenerstellung: {str(e)}",
            height=600,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
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
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Jahr', 'Zimmeranzahl_num', 'MedianPreis']
    if any(col not in df_quartier.columns for col in required_columns) or df_quartier.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Keine Daten verfügbar. Bitte führen Sie die Datenvorbereitungsskripte aus.",
            height=400,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    try:
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
                title=f"Keine Daten verfügbar für die ausgewählten Quartiere und {selected_zimmer} Zimmer",
                height=400,
                width=800,
                # Add dark theme
                plot_bgcolor=DARK_CARD_BG,
                paper_bgcolor=DARK_CARD_BG,
                font=dict(color=DARK_TEXT)
            )
            return fig
        
        # Calculate average prices per neighborhood
        df_grouped = df_filtered.groupby('Quartier').agg({
            'MedianPreis': 'mean',
            'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
        }).reset_index()
        
        # Sort neighborhoods by price
        df_grouped = df_grouped.sort_values('MedianPreis', ascending=False)
        
        # Create bar chart for MedianPreis
        fig = go.Figure()
        
        # MedianPreis bars
        fig.add_trace(go.Bar(
            x=df_grouped['Quartier'],
            y=df_grouped['MedianPreis'],
            name='Median-Kaufpreis (CHF)',
            marker_color=ZURICH_BLUE,
            text=df_grouped['MedianPreis'].apply(lambda x: f'{x:,.0f} CHF'),
            textposition='auto',
            textfont=dict(color=DARK_TEXT)
        ))
        
        # Adjust layout with dark theme
        fig.update_layout(
            title=f'Immobilienpreisvergleich ({neuestes_jahr}, {selected_zimmer} Zimmer)',
            xaxis_title='Quartier',
            yaxis_title='Preis (CHF)',
            height=400,
            width=800,
            barmode='group',
            xaxis={'categoryorder': 'total descending'},
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        
        # Update axes with dark theme
        fig.update_xaxes(
            tickfont=dict(color=DARK_TEXT),
            title_font=dict(color=DARK_TEXT),
            gridcolor=GRID_COLOR
        )
        
        fig.update_yaxes(
            tickfont=dict(color=DARK_TEXT),
            title_font=dict(color=DARK_TEXT),
            gridcolor=GRID_COLOR
        )
        
        return fig
    except Exception as e:
        print(f"Error creating price comparison chart: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler bei der Diagrammerstellung: {str(e)}",
            height=400,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
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
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Jahr', 'Zimmeranzahl_num', 'MedianPreis']
    if any(col not in df_quartier.columns for col in required_columns) or df_quartier.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Keine Daten verfügbar. Bitte führen Sie die Datenvorbereitungsskripte aus.",
            height=400,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig
    
    try:
        # Filter data for the selected neighborhoods and number of rooms
        df_filtered = df_quartier[
            (df_quartier['Zimmeranzahl_num'] == selected_zimmer) &
            (df_quartier['Quartier'].isin(selected_quartiere))
        ].copy()
        
        # If no data available, return empty figure
        if df_filtered.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"Keine Daten verfügbar für die ausgewählten Quartiere und {selected_zimmer} Zimmer",
                height=400,
                width=800,
                # Add dark theme
                plot_bgcolor=DARK_CARD_BG,
                paper_bgcolor=DARK_CARD_BG,
                font=dict(color=DARK_TEXT)
            )
            return fig
        
        # Group by year and neighborhood
        df_grouped = df_filtered.groupby(['Jahr', 'Quartier']).agg({
            'MedianPreis': 'mean',
            'PreisProQm': 'mean' if 'PreisProQm' in df_filtered.columns else lambda x: None
        }).reset_index()
        
        # Create plot
        # Use blue color palette
        blue_palette = px.colors.sequential.Blues[2:]  # Skip lightest blues for dark theme
        if len(blue_palette) < len(selected_quartiere):
            blue_palette = [ZURICH_BLUE] * len(selected_quartiere)  # Fallback to Zurich blue
            
        fig = px.line(
            df_grouped, 
            x='Jahr', 
            y='MedianPreis', 
            color='Quartier',
            markers=True,
            title=f'Preisentwicklung ({selected_zimmer} Zimmer)',
            height=400,
            width=800,
            color_discrete_sequence=blue_palette
        )
        
        # Adjust layout with dark theme
        fig.update_layout(
            xaxis_title='Jahr',
            yaxis_title='Median-Kaufpreis (CHF)',
            legend_title='Quartier',
            yaxis=dict(tickformat=',.0f'),
            hovermode='x unified',
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        
        # Update axes with dark theme
        fig.update_xaxes(
            tickfont=dict(color=DARK_TEXT),
            title_font=dict(color=DARK_TEXT),
            gridcolor=GRID_COLOR
        )
        
        fig.update_yaxes(
            tickfont=dict(color=DARK_TEXT),
            title_font=dict(color=DARK_TEXT),
            gridcolor=GRID_COLOR
        )
        
        # Update legend with dark theme
        fig.update_layout(
            legend=dict(
                font=dict(color=DARK_TEXT),
                bgcolor=DARK_CARD_BG,
                bordercolor=GRID_COLOR
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error creating price time series: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler bei der Diagrammerstellung: {str(e)}",
            height=400,
            width=800,
            # Add dark theme
            plot_bgcolor=DARK_CARD_BG,
            paper_bgcolor=DARK_CARD_BG,
            font=dict(color=DARK_TEXT)
        )
        return fig