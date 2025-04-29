"""
Maps module for Zurich Real Estate Price Prediction app.
Provides interactive map visualizations using plotly and folium.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

# Zurich neighborhood coordinates (approximate centers)
ZURICH_COORDINATES = {
    "Altstadt": [47.3723, 8.5423],
    "Escher Wyss": [47.3914, 8.5152],
    "Gewerbeschule": [47.3845, 8.5293],
    "Hochschulen": [47.3764, 8.5468],
    "Höngg": [47.4036, 8.4894],
    "Oerlikon": [47.4119, 8.5450],
    "Seebach": [47.4231, 8.5392],
    "Altstetten": [47.3889, 8.4833],
    "Albisrieden": [47.3783, 8.4893],
    "Sihlfeld": [47.3720, 8.5147],
    "Friesenberg": [47.3663, 8.5030],
    "Leimbach": [47.3407, 8.5124],
    "Wollishofen": [47.3510, 8.5301],
    "Enge": [47.3650, 8.5288],
    "Wiedikon": [47.3666, 8.5182],
    "Hard": [47.3845, 8.5090],
    "Unterstrass": [47.3887, 8.5400],
    "Oberstrass": [47.3860, 8.5487]
}

# Zurich city center coordinates
ZURICH_CENTER = [47.3769, 8.5417]

def get_plotly_map(neighborhood_prices):
    """
    Create a plotly.express scatter_mapbox visualization of Zurich real estate prices.
    
    Args:
        neighborhood_prices (dict): Dictionary mapping neighborhoods to prices
    
    Returns:
        plotly.graph_objects.Figure: Interactive map visualization
    """
    # Prepare data for plotly
    data = []
    for neighborhood, price in neighborhood_prices.items():
        if neighborhood in ZURICH_COORDINATES:
            data.append({
                "Neighborhood": neighborhood,
                "Price": price,
                "lat": ZURICH_COORDINATES[neighborhood][0],
                "lon": ZURICH_COORDINATES[neighborhood][1]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Normalize prices for marker size (so expensive areas have larger markers)
    max_price = df["Price"].max()
    min_price = df["Price"].min()
    df["size"] = 10 + 30 * (df["Price"] - min_price) / (max_price - min_price)
    
    # Create the map
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="Price",
        size="size",
        hover_name="Neighborhood",
        hover_data=["Price"],
        color_continuous_scale="Viridis",
        zoom=12,
        mapbox_style="carto-positron",
        title="Zurich Real Estate Prices by Neighborhood"
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Price (CHF)"
        )
    )
    
    return fig

def get_folium_map(neighborhood_prices):
    """
    Create a folium map visualization of Zurich real estate prices.
    
    Args:
        neighborhood_prices (dict): Dictionary mapping neighborhoods to prices
    
    Returns:
        folium.Map: Interactive map visualization
    """
    # Create a base map centered on Zurich
    m = folium.Map(location=ZURICH_CENTER, zoom_start=12, tiles="CartoDB positron")
    
    # Add markers for each neighborhood
    for neighborhood, price in neighborhood_prices.items():
        if neighborhood in ZURICH_COORDINATES:
            # Format price as string
            price_str = f"CHF {price:,.2f}"
            
            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial; min-width: 180px;">
                <h4>{neighborhood}</h4>
                <p><b>Price:</b> {price_str}</p>
            </div>
            """
            
            # Add marker
            folium.Marker(
                location=ZURICH_COORDINATES[neighborhood],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{neighborhood}: {price_str}",
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m)
    
    # Create heatmap data
    heatmap_data = []
    for neighborhood, price in neighborhood_prices.items():
        if neighborhood in ZURICH_COORDINATES:
            # Normalize price value for heatmap intensity
            max_price = max(neighborhood_prices.values())
            min_price = min(neighborhood_prices.values())
            intensity = (price - min_price) / (max_price - min_price)
            
            # Add to heatmap data
            heatmap_data.append(
                ZURICH_COORDINATES[neighborhood] + [intensity]
            )
    
    # Add heatmap layer
    HeatMap(heatmap_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def add_neighborhood_boundaries(m):
    """
    Add Zurich neighborhood boundaries to a folium map.
    This is a placeholder function - in a real implementation, you would
    load GeoJSON data for Zurich neighborhoods.
    
    Args:
        m (folium.Map): Folium map to add boundaries to
    """
    # In a real implementation, you would load GeoJSON data for Zurich neighborhoods
    # This would typically be done by loading a GeoJSON file:
    # with open("data/zurich_neighborhoods.geojson") as f:
    #     zurich_geojson = json.load(f)
    
    # For this example, we'll create a simple polygon for demonstration
    # This is just a placeholder - not actual boundaries
    for neighborhood, center in ZURICH_COORDINATES.items():
        # Create a simple polygon around the neighborhood center
        # In a real implementation, use actual GeoJSON boundaries
        points = []
        radius = 0.008  # Approximately 500-800 meters
        for angle in range(0, 360, 60):
            rad_angle = np.radians(angle)
            lat = center[0] + radius * np.cos(rad_angle)
            lon = center[1] + radius * np.sin(rad_angle)
            points.append([lat, lon])
        points.append(points[0])  # Close the polygon
        
        # Add the polygon to the map
        folium.Polygon(
            locations=points,
            popup=neighborhood,
            tooltip=neighborhood,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.2,
            weight=2
        ).add_to(m)
    
    return m

def display_plotly_map(neighborhood_prices):
    """Display an interactive plotly map in Streamlit"""
    fig = get_plotly_map(neighborhood_prices)
    st.plotly_chart(fig, use_container_width=True)

def display_folium_map(neighborhood_prices, show_heatmap=True):
    """Display an interactive folium map in Streamlit"""
    m = get_folium_map(neighborhood_prices)
    
    # Add neighborhood boundaries
    add_neighborhood_boundaries(m)
    
    # Display the map
    folium_static(m, width=800, height=600)

def display_map(neighborhood_prices, map_type="plotly"):
    """
    Display an interactive map in Streamlit.
    
    Args:
        neighborhood_prices (dict): Dictionary mapping neighborhoods to prices
        map_type (str): Type of map to display ('plotly' or 'folium')
    """
    # Map type selection
    map_options = ["Plotly Map", "Folium Map with Markers", "Folium Heatmap"]
    selected_map = st.radio("Select Map Type:", map_options, index=0)
    
    if selected_map == "Plotly Map":
        display_plotly_map(neighborhood_prices)
    elif selected_map == "Folium Map with Markers":
        display_folium_map(neighborhood_prices, show_heatmap=False)
    else:
        display_folium_map(neighborhood_prices, show_heatmap=True)

def predict_prices_for_all_neighborhoods(predict_price_func, room_count, building_age, travel_time):
    """
    Predict prices for all neighborhoods using the given prediction function.
    
    Args:
        predict_price_func: Function to predict price given parameters
        room_count (int): Number of rooms
        building_age (str): Building age category
        travel_time (int): Maximum travel time
    
    Returns:
        dict: Dictionary mapping neighborhoods to predicted prices
    """
    neighborhood_prices = {}
    for neighborhood in ZURICH_COORDINATES.keys():
        price = predict_price_func(neighborhood, room_count, building_age, travel_time)
        neighborhood_prices[neighborhood] = price
    
    return neighborhood_prices
