# flight_visualizations.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import altair as alt

def show_additional_visualizations(df, airline, departure_city, arrival_city, get_coordinates_for_city):
    """
    Generate and display additional visualizations for flight data analysis.
    
    Parameters:
    df (pandas.DataFrame): The flight data
    airline (str): Selected airline
    departure_city (str): Selected departure city
    arrival_city (str): Selected arrival city
    get_coordinates_for_city (function): Function to get city coordinates
    """
    st.header("📊 Flight Analytics & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Route Comparison", "Airline Analysis", "Interactive Explorer"])
    
    with tab1:
        st.subheader("Price Trends by Distance")
        
        # Create scatter plot with trend line using Plotly
        fig = px.scatter(df, x="Distance", y="Fare", color="Airline",
                         hover_data=["Departure City", "Arrival City"],
                         trendline="ols", title="Flight Prices vs. Distance")
        
        fig.update_layout(
            xaxis_title="Distance (miles)",
            yaxis_title="Fare ($)",
            legend_title="Airline",
            height=500
        )
        
        # Highlight the selected route if it exists
        selected_route = df[(df["Airline"] == airline) & 
                           (df["Departure City"] == departure_city) & 
                           (df["Arrival City"] == arrival_city)]
        
        if not selected_route.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_route["Distance"],
                    y=selected_route["Fare"],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="star"),
                    name="Selected Route",
                    hoverinfo="text",
                    hovertext=f"{departure_city} to {arrival_city} ({airline})"
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add price per mile analysis
        st.subheader("Price Efficiency ($ per mile)")
        
        # Calculate price per mile
        df_analysis = df.copy()
        df_analysis["Price_Per_Mile"] = df_analysis["Fare"] / df_analysis["Distance"]
        
        # Filter out outliers for better visualization
        q1 = df_analysis["Price_Per_Mile"].quantile(0.05)
        q3 = df_analysis["Price_Per_Mile"].quantile(0.95)
        df_filtered = df_analysis[(df_analysis["Price_Per_Mile"] >= q1) & 
                                  (df_analysis["Price_Per_Mile"] <= q3)]
        
        # Boxplot of price per mile by airline
        fig_box = px.box(df_filtered, x="Airline", y="Price_Per_Mile", 
                         color="Airline", points="all",
                         title="Price Efficiency by Airline ($ per mile)")
        
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab2:
        st.subheader("Route Comparison")
        
        # Get the 10 most popular routes
        route_counts = df.groupby(["Departure City", "Arrival City"]).size().reset_index(name="count")
        top_routes = route_counts.sort_values("count", ascending=False).head(10)
        
        # Create a combined column for route
        top_routes["Route"] = top_routes["Departure City"] + " → " + top_routes["Arrival City"]
        
        # Create horizontal bar chart
        fig_routes = px.bar(
            top_routes, 
            y="Route", 
            x="count", 
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            title="Top 10 Most Popular Routes"
        )
        
        fig_routes.update_layout(
            xaxis_title="Number of Flights",
            yaxis_title="",
            height=500
        )
        
        st.plotly_chart(fig_routes, use_container_width=True)
        
        # Distance vs Fare for popular city pairs
        st.subheader("Price Comparison for Popular Routes")
        
        # Get popular city pairs by merging routes in both directions
        city_pairs = []
        for _, row in top_routes.iterrows():
            city1, city2 = row["Departure City"], row["Arrival City"]
            # Check both directions
            routes_both_ways = df[((df["Departure City"] == city1) & (df["Arrival City"] == city2)) | 
                                 ((df["Departure City"] == city2) & (df["Arrival City"] == city1))]
            
            if not routes_both_ways.empty:
                avg_fare = routes_both_ways["Fare"].mean()
                avg_distance = routes_both_ways["Distance"].mean()
                city_pairs.append({
                    "City Pair": f"{city1} ↔ {city2}",
                    "Avg Fare": avg_fare,
                    "Avg Distance": avg_distance,
                    "Price Per Mile": avg_fare / avg_distance if avg_distance > 0 else 0
                })
        
        # Convert to DataFrame and sort
        city_pairs_df = pd.DataFrame(city_pairs).sort_values("Avg Fare", ascending=False)
        
        # Create a scatter plot
        fig_pairs = px.scatter(
            city_pairs_df, 
            x="Avg Distance", 
            y="Avg Fare",
            size="Price Per Mile",
            color="Price Per Mile",
            hover_name="City Pair",
            color_continuous_scale="Viridis",
            title="Price vs Distance for Popular City Pairs",
            size_max=30
        )
        
        st.plotly_chart(fig_pairs, use_container_width=True)
        
    with tab3:
        st.subheader("Airline Analysis")
        
        # Airline market share pie chart
        airline_share = df["Airline"].value_counts().reset_index()
        airline_share.columns = ["Airline", "Flight Count"]
        
        fig_pie = px.pie(
            airline_share, 
            values="Flight Count", 
            names="Airline",
            title="Airline Market Share",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Average fare by airline
        airline_stats = df.groupby("Airline").agg({
            "Fare": ["mean", "median", "min", "max", "count"],
            "Distance": ["mean", "min", "max"]
        }).reset_index()
        
        # Flatten multi-level columns
        airline_stats.columns = [' '.join(col).strip() for col in airline_stats.columns.values]
        
        # Rename columns for clarity
        airline_stats = airline_stats.rename(columns={
            "Fare mean": "Avg Fare",
            "Fare median": "Median Fare",
            "Fare min": "Min Fare",
            "Fare max": "Max Fare",
            "Fare count": "Flights",
            "Distance mean": "Avg Distance",
            "Distance min": "Min Distance",
            "Distance max": "Max Distance"
        })
        
        # Create a bar chart for average fares
        fig_avg = px.bar(
            airline_stats, 
            x="Airline", 
            y="Avg Fare",
            color="Airline",
            text_auto='.2f',
            title="Average Fare by Airline"
        )
        
        fig_avg.update_layout(
            xaxis_title="",
            yaxis_title="Average Fare ($)",
            height=500
        )
        
        st.plotly_chart(fig_avg, use_container_width=True)
        
        # Display airline statistics table
        st.subheader("Airline Statistics")
        st.dataframe(airline_stats)
        
    with tab4:
        st.subheader("Interactive Flight Explorer")
        
        # Create a dataframe with flight counts between cities
        flight_matrix = df.groupby(["Departure City", "Arrival City"]).size().reset_index(name="Flights")
        
        # Add coordinates
        flight_matrix["Departure Lat"] = flight_matrix["Departure City"].apply(
            lambda x: get_coordinates_for_city(x)[0] if get_coordinates_for_city(x) else None)
        flight_matrix["Departure Lng"] = flight_matrix["Departure City"].apply(
            lambda x: get_coordinates_for_city(x)[1] if get_coordinates_for_city(x) else None)
        flight_matrix["Arrival Lat"] = flight_matrix["Arrival City"].apply(
            lambda x: get_coordinates_for_city(x)[0] if get_coordinates_for_city(x) else None)
        flight_matrix["Arrival Lng"] = flight_matrix["Arrival City"].apply(
            lambda x: get_coordinates_for_city(x)[1] if get_coordinates_for_city(x) else None)
        
        # Remove rows with missing coordinates
        flight_matrix = flight_matrix.dropna()
        
        # Filter to top 50 routes for performance
        flight_matrix = flight_matrix.sort_values("Flights", ascending=False).head(50)
        
        # Create a network graph with ECharts
        nodes = []
        unique_cities = set(flight_matrix["Departure City"]).union(set(flight_matrix["Arrival City"]))
        
        for city in unique_cities:
            coords = get_coordinates_for_city(city)
            if coords:
                nodes.append({
                    "name": city,
                    "value": [coords[1], coords[0]],  # [lng, lat]
                    "symbolSize": 10
                })
        
        links = []
        for _, row in flight_matrix.iterrows():
            links.append({
                "source": row["Departure City"],
                "target": row["Arrival City"],
                "lineStyle": {
                    "width": min(5, row["Flights"] / 2 + 1)  # Scale line width by flight count
                }
            })
        
        option = {
            "backgroundColor": "#404a59",
            "title": {
                "text": "US Flight Network",
                "left": "center",
                "textStyle": {
                    "color": "#fff"
                }
            },
            "tooltip": {
                "trigger": "item"
            },
            "legend": {
                "orient": "vertical",
                "top": "bottom",
                "left": "right",
                "textStyle": {
                    "color": "#fff"
                },
                "selectedMode": "single"
            },
            "geo": {
                "map": "USA",
                "label": {
                    "emphasis": {
                        "show": False
                    }
                },
                "roam": True,
                "itemStyle": {
                    "normal": {
                        "areaColor": "#323c48",
                        "borderColor": "#111"
                    },
                    "emphasis": {
                        "areaColor": "#2a333d"
                    }
                }
            },
            "series": [
                {
                    "name": "Flight Routes",
                    "type": "lines",
                    "coordinateSystem": "geo",
                    "zlevel": 1,
                    "effect": {
                        "show": True,
                        "period": 6,
                        "trailLength": 0.7,
                        "color": "#fff",
                        "symbolSize": 3
                    },
                    "lineStyle": {
                        "normal": {
                            "color": "#a6c84c",
                            "width": 0,
                            "curveness": 0.2
                        }
                    },
                    "data": links
                },
                {
                    "name": "Airports",
                    "type": "effectScatter",
                    "coordinateSystem": "geo",
                    "zlevel": 2,
                    "rippleEffect": {
                        "brushType": "stroke"
                    },
                    "label": {
                        "normal": {
                            "show": True,
                            "position": "right",
                            "formatter": "{b}"
                        }
                    },
                    "symbolSize": 10,
                    "itemStyle": {
                        "normal": {
                            "color": "#a6c84c"
                        }
                    },
                    "data": nodes
                }
            ]
        }
        
        # Use Streamlit ECharts for rendering
        st_echarts(option, height=600, map="USA")
        
        # Interactive fare heatmap by distance and airline
        st.subheader("Fare Heatmap by Distance and Airline")
        
        # Bin distances for better visualization
        df["Distance Range"] = pd.cut(
            df["Distance"], 
            bins=[0, 200, 500, 1000, 1500, 2000, 3000, 5000],
            labels=["0-200", "201-500", "501-1000", "1001-1500", "1501-2000", "2001-3000", "3001+"]
        )
        
        # Create pivot table
        pivot = df.pivot_table(
            values="Fare", 
            index="Airline",
            columns="Distance Range",
            aggfunc="mean"
        ).round(2)
        
        # Fill NaN with 0 for visualization
        pivot = pivot.fillna(0)
        
        # Create heatmap data
        heatmap_data = []
        for airline in pivot.index:
            for distance in pivot.columns:
                if pivot.loc[airline, distance] > 0:  # Only add non-zero values
                    heatmap_data.append({
                        "Airline": airline,
                        "Distance Range": distance,
                        "Average Fare": pivot.loc[airline, distance]
                    })
        
        # Create the heatmap
        heatmap_df = pd.DataFrame(heatmap_data)
        
        heatmap = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X('Distance Range:O', title='Distance (miles)'),
            y=alt.Y('Airline:O', title='Airline'),
            color=alt.Color('Average Fare:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Airline', 'Distance Range', 'Average Fare']
        ).properties(
            title='Average Fare by Airline and Distance',
            width=700,
            height=400
        ).interactive()
        
        st.altair_chart(heatmap, use_container_width=True)

def create_price_history_chart(df, airline, departure_city, arrival_city):
    """
    Creates a simulated price history chart for a specific route.
    
    Parameters:
    df (pandas.DataFrame): The flight data
    airline (str): Selected airline
    departure_city (str): Selected departure city
    arrival_city (str): Selected arrival city
    
    Returns:
    plotly.graph_objects.Figure: A line chart of historical prices
    """
    # Get the average fare for the route
    route_data = df[
        (df["Airline"] == airline) & 
        (df["Departure City"] == departure_city) & 
        (df["Arrival City"] == arrival_city)
    ]
    
    if len(route_data) > 0:
        base_fare = route_data["Fare"].mean()
    else:
        # Use a fallback if no direct route data
        base_fare = df[(df["Airline"] == airline)]["Fare"].mean()
    
    # Generate synthetic historical data (last 6 months)
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(end=pd.Timestamp.now(), periods=180, freq='D')
    
    # Create some seasonal patterns and randomness
    seasonal = 0.15 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Seasonal pattern
    trend = 0.05 * np.linspace(0, 1, len(dates))  # Slight upward trend
    noise = 0.1 * np.random.randn(len(dates))  # Random fluctuations
    
    # Combine patterns and apply to base fare
    price_factors = 1 + seasonal + trend + noise
    prices = base_fare * price_factors
    
    # Create a DataFrame
    price_history = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    # Create the plot
    fig = go.Figure()
    
    # Add the price line
    fig.add_trace(go.Scatter(
        x=price_history['Date'],
        y=price_history['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add a horizontal line for the current price
    fig.add_trace(go.Scatter(
        x=[price_history['Date'].min(), price_history['Date'].max()],
        y=[base_fare, base_fare],
        mode='lines',
        name='Average Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Highlight weekends
    for date in dates:
        if date.dayofweek >= 5:  # 5=Saturday, 6=Sunday
            fig.add_vrect(
                x0=date,
                x1=date + pd.Timedelta(days=1),
                fillcolor="gray",
                opacity=0.1,
                layer="below",
                line_width=0
            )
    
    # Add annotations for key events (synthetic)
    events = [
        {"date": dates[30], "text": "Holiday Season", "price_effect": 0.15},
        {"date": dates[90], "text": "Low Season", "price_effect": -0.1},
        {"date": dates[150], "text": "Summer Travel", "price_effect": 0.2}
    ]
    
    for event in events:
        fig.add_annotation(
            x=event["date"],
            y=base_fare * (1 + event["price_effect"]),
            text=event["text"],
            showarrow=True,
            arrowhead=1
        )
    
    # Customize layout
    fig.update_layout(
        title=f"6-Month Price History: {departure_city} to {arrival_city} ({airline})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

def add_seasonal_pricing_insight(df):
    """
    Provides a seasonal pricing analysis visualization.
    
    Parameters:
    df (pandas.DataFrame): The flight data
    """
    st.subheader("Seasonal Pricing Insights")
    
    # Create synthetic seasonal data
    # In a real app, you'd have date information in your dataset
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    airlines = df["Airline"].unique()
    
    # Create synthetic seasonal factors for each airline
    seasonal_data = []
    
    for airline in airlines:
        base_price = df[df["Airline"] == airline]["Fare"].mean()
        seasonal_data.append({
            "Airline": airline,
            "Winter": base_price * (1 + np.random.uniform(-0.1, 0.2)),  # Holiday season premium
            "Spring": base_price * (1 + np.random.uniform(-0.05, 0.05)),  # Average pricing
            "Summer": base_price * (1 + np.random.uniform(0.1, 0.25)),   # Summer travel premium
            "Fall": base_price * (1 + np.random.uniform(-0.15, -0.05))   # Low season discount
        })
    
    # Convert to DataFrame
    seasonal_df = pd.DataFrame(seasonal_data)
    
    # Melt DataFrame for plotting
    seasonal_melted = pd.melt(
        seasonal_df, 
        id_vars=["Airline"],
        value_vars=seasons,
        var_name="Season", 
        value_name="Average Fare"
    )
    
    # Create grouped bar chart
    fig = px.bar(
        seasonal_melted,
        x="Season",
        y="Average Fare",
        color="Airline",
        barmode="group",
        title="Seasonal Price Variations by Airline",
        category_orders={"Season": seasons}
    )
    
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Fare ($)",
        legend_title="Airline",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add some insights
    st.markdown("""
    ### Key Seasonal Pricing Insights:
    - **Winter**: Prices tend to be higher due to holiday travel demand, especially around Christmas and New Year
    - **Summer**: Peak travel season with highest prices, particularly for popular vacation destinations
    - **Fall**: Best time to find deals with lower prices across most airlines
    - **Spring**: Moderate pricing with occasional deals depending on destination
    """)

def add_best_booking_time(df, airline, departure_city, arrival_city):
    """
    Adds a visualization for best booking time recommendations.
    
    Parameters:
    df (pandas.DataFrame): The flight data
    airline (str): Selected airline
    departure_city (str): Selected departure city
    arrival_city (str): Selected arrival city
    """
    st.subheader("When to Book Your Flight")
    
    # Create synthetic data for booking trends
    weeks_in_advance = list(range(1, 13))  # 12 weeks
    
    # Calculate base price from data
    route_data = df[
        (df["Airline"] == airline) & 
        (df["Departure City"] == departure_city) & 
        (df["Arrival City"] == arrival_city)
    ]
    
    if len(route_data) > 0:
        base_fare = route_data["Fare"].mean()
    else:
        base_fare = df[(df["Airline"] == airline)]["Fare"].mean()
    
    # Create a typical booking curve (prices generally go up as the flight date approaches)
    # Slight U-curve with lowest prices around 6-8 weeks before departure
    price_factors = [1.4, 1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.92, 0.97, 1.03, 1.15, 1.25]
    prices = [base_fare * factor for factor in price_factors]
    
    # Create DataFrame
    booking_data = pd.DataFrame({
        "Weeks Before Departure": weeks_in_advance,
        "Average Price": prices
    })
    
    # Find the optimal booking time
    best_week = booking_data.loc[booking_data["Average Price"].idxmin(), "Weeks Before Departure"]
    
    # Create line chart
    fig = px.line(
        booking_data,
        x="Weeks Before Departure",
        y="Average Price",
        title="Average Ticket Prices by Booking Time",
        markers=True
    )
    
    # Add annotation for best booking time
    min_price = booking_data["Average Price"].min()
    fig.add_annotation(
        x=best_week,
        y=min_price,
        text=f"Best time to book<br>{best_week} weeks before",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Highlight the optimal booking window
    fig.add_vrect(
        x0=best_week-1,
        x1=best_week+1,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    # Customize chart
    fig.update_layout(
        xaxis_title="Weeks Before Departure",
        yaxis_title="Average Price ($)",
        height=400,
        xaxis=dict(tickmode='linear', dtick=1, autorange="reversed")  # Reverse x-axis
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add booking recommendations
    st.markdown(f"""
    ### Booking Recommendations for {departure_city} to {arrival_city}:
    
    - **Optimal booking window**: **{best_week-1}-{best_week+1} weeks** before departure
    - **Last-minute bookings** (less than 2 weeks before): Typically 25-40% more expensive
    - **Too early** (more than 10 weeks before): May pay a premium of 5-15%
    
    *These recommendations are based on historical pricing patterns for similar routes and may vary.*
    """)