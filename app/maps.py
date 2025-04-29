import pandas as pd
import numpy as np
import plotly.express as px

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
    # Add more neighborhoods as needed
}

# Zurich center coordinates
ZURICH_CENTER = (47.3769, 8.5417)
ZURICH_ZOOM = 12

def add_coordinates_to_data(data):
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
    for neighborhood, (lat, lon) in ZURICH_COORDS.items():
        mask = result['neighborhood'] == neighborhood
        result.loc[mask, 'lat'] = lat
        result.loc[mask, 'lon'] = lon
    
    # For neighborhoods without specific coordinates, use Zurich center
    result['lat'].fillna(ZURICH_CENTER[0], inplace=True)
    result['lon'].fillna(ZURICH_CENTER[1], inplace=True)
    
    return result

def create_price_heatmap(data):
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
        title="Property Prices by Neighborhood"
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Price (CHF)"
        )
    )
    
    return fig

def create_travel_time_map(origin, destinations, travel_times):
    """
    Create a map showing travel times from origin to destinations
    
    Args:
        origin: Origin neighborhood name
        destinations: List of destination names
        travel_times: Dictionary mapping destinations to travel times
        
    Returns:
        plotly.Figure: Plotly map figure
    """
    # This is a placeholder function
    # In a real implementation, this would create a map with travel time visualizations
    
    # Create a simple scatter map for now
    fig = px.scatter_mapbox(
        pd.DataFrame({
            'location': [origin] + destinations,
            'lat': [ZURICH_CENTER[0]] * (len(destinations) + 1),
            'lon': [ZURICH_CENTER[1]] * (len(destinations) + 1),
            'type': ['Origin'] + ['Destination'] * len(destinations)
        }),
        lat='lat',
        lon='lon',
        color='type',
        hover_name='location',
        zoom=ZURICH_ZOOM,
        mapbox_style="open-street-map",
        title="Travel Time Map"
    )
    
    return fig