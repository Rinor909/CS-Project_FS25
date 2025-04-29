"""
Maps Module for Zurich Real Estate Price Prediction App
------------------------------------------------------
Purpose: Create interactive maps for visualizing property prices in Zurich

Tasks:
1. Generate heatmaps of property prices across Zurich
2. Display travel time radii from key destinations
3. Create interactive maps with filtering options

Owner: Matthieu (Primary), Anna (Support)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_price_heatmap(neighborhood_df, neighborhood_coords, price_column='HAMedianPreis'):
    """
    Create a heatmap of property prices across Zurich neighborhoods.
    
    Parameters:
    - neighborhood_df: DataFrame with neighborhood price data
    - neighborhood_coords: Dictionary mapping neighborhoods to coordinates
    - price_column: Column name containing price data
    
    Returns:
    - Plotly figure object
    """
    logger.info("Creating price heatmap")
    
    # Extract data for plotting
    data = []
    
    # This is a placeholder implementation
    # In the real implementation, you would:
    # 1. Aggregate neighborhood data to get median prices
    # 2. Map each neighborhood to its coordinates
    # 3. Create a proper heatmap
    
    # For now, we'll create a sample map with dummy data
    for neighborhood, coords in neighborhood_coords.items():
        # Get median price for this neighborhood
        # In a real implementation, you would filter the dataframe to get the actual price
        median_price = 1000000  # Placeholder value
        
        data.append({
            'neighborhood': neighborhood,
            'lat': coords['lat'],
            'lon': coords['lng'],
            'price': median_price
        })
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(data)
    
    # Create map
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='price',
        size='price',
        size_max=15,
        zoom=11,
        center={"lat": 47.3769, "lon": 8.5417},  # Center of Zurich
        mapbox_style="carto-positron",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'price': 'Price (CHF)'},
        hover_name='neighborhood',
        hover_data={'neighborhood': False, 'price': True, 'lat': False, 'lon': False}
    )
    
    fig.update_layout(
        title="Property Prices in Zurich",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

def create_travel_time_map(travel_time_df, neighborhood_coords, destination_coords, destination_name):
    """
    Create a map showing travel times from a specific destination.
    
    Parameters:
    - travel_time_df: DataFrame with travel time data
    - neighborhood_coords: Dictionary mapping neighborhoods to coordinates
    - destination_coords: Dictionary with destination coordinates
    - destination_name: Name of the destination
    
    Returns:
    - Plotly figure object
    """
    logger.info(f"Creating travel time map for {destination_name}")
    
    # Extract data for plotting
    data = []
    
    # This is a placeholder implementation
    # In the real implementation, you would:
    # 1. Filter travel time data for the specific destination
    # 2. Map each neighborhood to its coordinates
    # 3. Create a travel time visualization
    
    # For now, we'll create a sample map with dummy data
    for neighborhood, coords in neighborhood_coords.items():
        # Get travel time for this neighborhood to the destination
        # In a real implementation, you would filter the dataframe to get the actual travel time
        travel_time = np.random.randint(5, 60)  # Placeholder value
        
        data.append({
            'neighborhood': neighborhood,
            'lat': coords['lat'],
            'lon': coords['lng'],
            'travel_time': travel_time
        })
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(data)
    
    # Create map
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='travel_time',
        size='travel_time',
        size_max=15,
        zoom=11,
        center={"lat": 47.3769, "lon": 8.5417},  # Center of Zurich
        mapbox_style="carto-positron",
        color_continuous_scale=px.colors.sequential.Viridis_r,  # Reverse colorscale so shorter times are greener
        labels={'travel_time': 'Time (min)'},
        hover_name='neighborhood',
        hover_data={'neighborhood': False, 'travel_time': True, 'lat': False, 'lon': False}
    )
    
    # Add destination marker
    fig.add_trace(
        go.Scattermapbox(
            lat=[destination_coords['lat']],
            lon=[destination_coords['lng']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            name=destination_name,
            hoverinfo='name'
        )
    )
    
    fig.update_layout(
        title=f"Travel Time to {destination_name}",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

def create_combined_map(neighborhood_df, travel_time_df, neighborhood_coords, destination_coords):
    """
    Create a combined map showing property prices and travel times.
    
    Parameters:
    - neighborhood_df: DataFrame with neighborhood price data
    - travel_time_df: DataFrame with travel time data
    - neighborhood_coords: Dictionary mapping neighborhoods to coordinates
    - destination_coords: Dictionary with destination coordinates
    
    Returns:
    - Plotly figure object
    """
    logger.info("Creating combined price and travel time map")
    
    # Create a figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Property Prices", "Travel Times"),
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]]
    )
    
    # Add price heatmap to first subplot
    price_fig = create_price_heatmap(neighborhood_df, neighborhood_coords)
    for trace in price_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add travel time map to second subplot
    # Let's use Hauptbahnhof as an example destination
    travel_time_fig = create_travel_time_map(
        travel_time_df,
        neighborhood_coords,
        destination_coords['Hauptbahnhof'],
        'Hauptbahnhof'
    )
    for trace in travel_time_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=11,
            center={"lat": 47.3769, "lon": 8.5417}  # Center of Zurich
        ),
        mapbox2=dict(
            style="carto-positron",
            zoom=11,
            center={"lat": 47.3769, "lon": 8.5417}  # Center of Zurich
        ),
        height=500,
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

def create_interactive_map(neighborhood_df, travel_time_df, neighborhood_coords, destination_coords, 
                          selected_neighborhood=None, max_travel_time=None):
    """
    Create an interactive map with filtering options.
    
    Parameters:
    - neighborhood_df: DataFrame with neighborhood price data
    - travel_time_df: DataFrame with travel time data
    - neighborhood_coords: Dictionary mapping neighborhoods to coordinates
    - destination_coords: Dictionary with destination coordinates
    - selected_neighborhood: Optionally filter to a specific neighborhood
    - max_travel_time: Optionally filter to areas within a certain travel time
    
    Returns:
    - Plotly figure object
    """
    logger.info("Creating interactive map")
    
    # This is a placeholder implementation
    # In a real implementation, you would:
    # 1. Filter data based on selected_neighborhood and max_travel_time
    # 2. Create a more sophisticated visualization
    
    # For now, we'll just create a simple price heatmap
    fig = create_price_heatmap(neighborhood_df, neighborhood_coords)
    
    # Add key destinations
    dest_lats = [coords['lat'] for coords in destination_coords.values()]
    dest_lons = [coords['lng'] for coords in destination_coords.values()]
    dest_names = list(destination_coords.keys())
    
    fig.add_trace(
        go.Scattermapbox(
            lat=dest_lats,
            lon=dest_lons,
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle'
            ),
            name='Key Destinations',
            text=dest_names,
            hoverinfo='text'
        )
    )
    
    # Highlight selected neighborhood if specified
    if selected_neighborhood and selected_neighborhood in neighborhood_coords:
        coords = neighborhood_coords[selected_neighborhood]
        fig.add_trace(
            go.Scattermapbox(
                lat=[coords['lat']],
                lon=[coords['lng']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='blue',
                    symbol='circle',
                    opacity=0.7
                ),
                name=selected_neighborhood,
                hoverinfo='name'
            )
        )
    
    # Add travel time radius if specified
    if max_travel_time and selected_neighborhood:
        # In a real implementation, you would calculate
        # the actual travel time radius based on the data
        # For now, we'll just add a circle as a placeholder
        coords = neighborhood_coords[selected_neighborhood]
        
        # Convert travel time to an approximate radius in meters
        # Assuming roughly 5 km/h walking speed
        radius = max_travel_time / 60 * 5 * 1000
        
        fig.add_trace(
            go.Scattermapbox(
                lat=[coords['lat']],
                lon=[coords['lng']],
                mode='markers',
                marker=dict(
                    size=radius/20,  # Scale down for visualization
                    color='rgba(0, 0, 255, 0.2)',
                    symbol='circle'
                ),
                name=f"{max_travel_time} min radius",
                hoverinfo='name'
            )
        )
    
    fig.update_layout(
        title="Zurich Property Prices and Travel Times",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig
