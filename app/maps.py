import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any

# Approximate Zurich neighborhood coordinates
# This is a placeholder - in a real app, you would have actual geocoordinates for each neighborhood
ZURICH_COORDS = {
    "Zurich (Kreis 1)": (47.3769, 8.5417),
    "Zurich (Kreis 2)": (47.3600, 8.5200),
    "Zurich (Kreis 3)": (47.3786, 8.5124),
    "Zurich (Kreis 4)": (47.3744, 8.5289),
    "Zurich (Kreis 5)": (47.3853, 8.5253),
    "Zurich (Kreis 6)": (47.3894, 8.5500),
    "Zurich (Kreis 7)": (47.3672, 8.5681),
    "Zurich (Kreis 8)": (47.3500, 8.5700),
    "Zurich (Kreis 9)": (47.3900, 8.4900),
    "Zurich (Kreis 10)": (47.4100, 8.5100),
    "Zurich (Kreis 11)": (47.4200, 8.5300),
    "Zurich (Kreis 12)": (47.3950, 8.5800),
    # Some additional areas in the greater Zurich region
    "Zurich-Altstetten": (47.3911, 8.4889),
    "Zurich-Oerlikon": (47.4108, 8.5456),
    "Zurich-Seebach": (47.4225, 8.5389),
    "Zurich-Affoltern": (47.4267, 8.5125),
    "Zurich-Schwamendingen": (47.4033, 8.5681),
    "Zurich-Fluntern": (47.3836, 8.5633),
    "Zurich-Hottingen": (47.3706, 8.5586),
    "Zurich-Wipkingen": (47.3919, 8.5250),
    "Zurich-Albisrieden": (47.3781, 8.4803),
    "Zurich-Leimbach": (47.3458, 8.5064),
    "Zurich-Wollishofen": (47.3458, 8.5303),
    "Zurich-Enge": (47.3625, 8.5308),
    "Zurich-Wiedikon": (47.3667, 8.5100),
    "Zurich-Friesenberg": (47.3603, 8.5014),
    "Zurich-Höngg": (47.4022, 8.4969),
}

# Key destinations in Zurich
KEY_DESTINATIONS = {
    "Hauptbahnhof": (47.3783, 8.5402),
    "ETH Zurich": (47.3763, 8.5475),
    "Zurich Airport": (47.4502, 8.5618),
    "Bahnhofstrasse": (47.3708, 8.5392),
}

# Zurich center coordinates and default zoom level
ZURICH_CENTER = (47.3769, 8.5417)
ZURICH_ZOOM = 12

def add_coordinates_to_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude and longitude to the data based on neighborhood
    
    Args:
        data: DataFrame with a 'neighborhood' column
        
    Returns:
        DataFrame: Data with added lat/lon columns
    """
    result = data.copy()
    
    # Add placeholder lat and lon columns
    result['lat'] = np.nan
    result['lon'] = np.nan
    
    # Fill in coordinates based on neighborhood
    for neighborhood in result['neighborhood'].unique():
        # Try exact match
        if neighborhood in ZURICH_COORDS:
            lat, lon = ZURICH_COORDS[neighborhood]
            mask = result['neighborhood'] == neighborhood
            result.loc[mask, 'lat'] = lat
            result.loc[mask, 'lon'] = lon
        else:
            # Try partial match
            for coord_name, (lat, lon) in ZURICH_COORDS.items():
                if coord_name in neighborhood or neighborhood in coord_name:
                    mask = result['neighborhood'] == neighborhood
                    result.loc[mask, 'lat'] = lat
                    result.loc[mask, 'lon'] = lon
                    break
    
    # For neighborhoods without specific coordinates, use Zurich center
    result['lat'].fillna(ZURICH_CENTER[0], inplace=True)
    result['lon'].fillna(ZURICH_CENTER[1], inplace=True)
    
    return result

def create_price_heatmap(data: pd.DataFrame) -> go.Figure:
    """
    Create a price heatmap using plotly
    
    Args:
        data: DataFrame with neighborhood and price data
        
    Returns:
        plotly.Figure: Plotly map figure
    """
    # Add coordinates to the data
    map_data = add_coordinates_to_data(data)
    
    # Group by neighborhood and calculate average price
    map_data = map_data.groupby('neighborhood').agg({
        'median_price': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    # Create the map
    fig = px.scatter_mapbox(
        map_data,
        lat='lat',
        lon='lon',
        color='median_price',
        size='median_price',
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15,
        zoom=ZURICH_ZOOM,
        mapbox_style="open-street-map",
        hover_name='neighborhood',
        hover_data={
            'lat': False,
            'lon': False,
            'median_price': True
        },
        labels={'median_price': 'Median Price (CHF)'},
        title="Property Prices by Neighborhood"
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Price (CHF)"
        ),
        title={
            'text': "Property Prices Across Zurich",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig

def create_travel_time_map(
    origin: str, 
    destinations: List[str], 
    travel_times: Dict[str, int]
) -> go.Figure:
    """
    Create a map showing travel times from origin to destinations
    
    Args:
        origin: Origin neighborhood name
        destinations: List of destination names
        travel_times: Dictionary mapping destinations to travel times
        
    Returns:
        plotly.Figure: Plotly map figure
    """
    # Get origin coordinates
    origin_lat, origin_lon = ZURICH_COORDS.get(origin, ZURICH_CENTER)
    
    # Prepare data for the map
    map_data = []
    
    # Add origin point
    map_data.append({
        'location': origin,
        'lat': origin_lat,
        'lon': origin_lon,
        'type': 'Origin',
        'travel_time': 0
    })
    
    # Add destination points
    for dest in destinations:
        dest_lat, dest_lon = KEY_DESTINATIONS.get(dest, ZURICH_CENTER)
        travel_time = travel_times.get(dest, 0)
        
        map_data.append({
            'location': dest,
            'lat': dest_lat,
            'lon': dest_lon,
            'type': 'Destination',
            'travel_time': travel_time
        })
    
    # Create DataFrame for plotting
    df = pd.DataFrame(map_data)
    
    # Create map with scatter plot
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='type',
        hover_name='location',
        hover_data={
            'lat': False,
            'lon': False,
            'travel_time': True,
            'type': True
        },
        zoom=11,
        mapbox_style="open-street-map",
        title="Travel Times from Selected Neighborhood"
    )
    
    # Add lines connecting origin to destinations
    for dest in destinations:
        dest_lat, dest_lon = KEY_DESTINATIONS.get(dest, ZURICH_CENTER)
        
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=[origin_lon, dest_lon],
            lat=[origin_lat, dest_lat],
            line=dict(width=2, color='red'),
            name=f"To {dest}",
            hoverinfo='text',
            text=f"To {dest}: {travel_times.get(dest, 0)} min"
        ))
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title={
            'text': f"Travel Routes from {origin}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def create_choropleth_map(data: pd.DataFrame, year: int) -> go.Figure:
    """
    Create a choropleth map of Zurich neighborhoods
    
    Note: This is a placeholder function - in a real app, you would need GeoJSON data for Zurich neighborhoods
    
    Args:
        data: DataFrame with neighborhood price data
        year: Year to filter data for
        
    Returns:
        plotly.Figure: Plotly choropleth map
    """
    # In a real implementation, this would use GeoJSON data for Zurich neighborhoods
    # This is just a placeholder that creates a scatter map instead
    
    # Filter data for the selected year
    year_data = data[data['year'] == year]
    
    # Add coordinates
    map_data = add_coordinates_to_data(year_data)
    
    # Group by neighborhood
    map_data = map_data.groupby('neighborhood').agg({
        'median_price': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    # Create scatter map (placeholder for choropleth)
    fig = px.scatter_mapbox(
        map_data,
        lat='lat',
        lon='lon',
        color='median_price',
        size='median_price',
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=15,
        zoom=ZURICH_ZOOM,
        mapbox_style="open-street-map",
        hover_name='neighborhood',
        hover_data={
            'lat': False,
            'lon': False,
            'median_price': True
        },
        title=f"Property Prices by Neighborhood ({year})"
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title={
            'text': f"Property Prices by Neighborhood ({year})",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig
