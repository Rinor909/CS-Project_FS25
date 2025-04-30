"""
Zürich Real Estate Price Prediction App

This Streamlit app predicts real estate prices in Zurich based on property
characteristics: location, building age, room count, and travel time to key destinations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import folium
import random
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Page configuration
st.set_page_config(
    page_title="Zürich Real Estate Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# Zurich Map Center Coordinates
ZURICH_COORDS = [47.3769, 8.5417]

# Function to load data based on availability
@st.cache_data
def load_data():
    """Load the real estate dataset based on what's available"""
    data_paths = [
        'data/processed/zuerich_immobilien_stichprobe.csv',  # First try the optimized sample
        'data/processed/zuerich_immobilien_komplett.csv',    # Then the complete dataset
        'data/processed/zuerich_immobilien_demo.csv'         # Finally the demo dataset
    ]
    
    # Try loading each dataset in order of preference
    for path in data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            print(f"Loaded dataset: {path} with {len(data)} rows")
            return data
    
    # If no data found, display error and return empty dataframe
    st.error("Keine Daten gefunden! Bitte führen Sie zuerst das Script 'datenbereinigung.py' aus.")
    return pd.DataFrame()

# Function to load statistics
@st.cache_data
def load_statistics():
    """Load statistics about the dataset"""
    stats_path = 'data/processed/statistiken.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    else:
        return {}

# Function to load or train a model
@st.cache_resource
def get_model(data):
    """Load or train a price prediction model"""
    model_path = 'data/processed/price_model.pkl'
    
    # If model exists, load it
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            return model
    
    # Otherwise train a new model
    if len(data) == 0:
        return None
    
    # Basic features for model
    features = ['Jahr']
    
    # Add room count encoding
    room_mapping = {
        '1 Zimmer': 1,
        '2 Zimmer': 2,
        '3 Zimmer': 3,
        '4 Zimmer': 4,
        '5+ Zimmer': 5
    }
    
    data['Zimmer_Num'] = data['Zimmeranzahl'].map(room_mapping).fillna(3)
    features.append('Zimmer_Num')
    
    # Add travel times if available
    travel_features = [col for col in data.columns if 'Reisezeit_' in col]
    features.extend(travel_features)
    
    # One-hot encode categorical features
    X = pd.get_dummies(data[features + ['Quartier', 'Baualter']], 
                       columns=['Quartier', 'Baualter'], drop_first=True)
    y = data['GeschaetzterPreis']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Function to load Zurich neighborhood coordinates
@st.cache_data
def load_neighborhood_coords():
    """Load or generate coordinates for Zurich neighborhoods"""
    coords_path = 'data/processed/quartier_coords.json'
    
    if os.path.exists(coords_path):
        with open(coords_path, 'r') as f:
            return json.load(f)
    
    # Simple placeholder coordinates for common Zurich neighborhoods
    # In a real app, we would use proper GeoJSON data
    zurich_center = [47.3769, 8.5417]
    
    # Generate random coordinates around Zurich center for neighborhoods
    neighborhood_coords = {}
    
    # Load neighborhood list
    quartiere_file = 'data/processed/quartier_liste.csv'
    if os.path.exists(quartiere_file):
        quartiere_df = pd.read_csv(quartiere_file)
        neighborhoods = quartiere_df['Quartier'].tolist()
    else:
        # Some Zurich neighborhoods as fallback
        neighborhoods = [
            "Enge", "Wollishofen", "Leimbach", "Adliswil", "Kilchberg", 
            "Rüschlikon", "Thalwil", "Oberrieden", "Horgen", "Affoltern", 
            "Oerlikon", "Seebach", "Schwamendingen", "Altstetten", "Albisrieden", 
            "City", "Lindenhof", "Rathaus", "Hochschulen", "Bellevue", "Seefeld"
        ]
    
    # Fixed coordinates for some key neighborhoods
    fixed_coords = {
        "City": [47.3744, 8.5410],
        "Oerlikon": [47.4114, 8.5442],
        "Altstetten": [47.3909, 8.4848],
        "Seefeld": [47.3583, 8.5550],
        "Enge": [47.3642, 8.5306]
    }
    
    # Generate coordinates for each neighborhood
    for neighborhood in neighborhoods:
        if neighborhood in fixed_coords:
            neighborhood_coords[neighborhood] = fixed_coords[neighborhood]
        else:
            # Random offset from center
            lat_offset = random.uniform(-0.02, 0.02)
            lng_offset = random.uniform(-0.02, 0.02)
            neighborhood_coords[neighborhood] = [
                zurich_center[0] + lat_offset, 
                zurich_center[1] + lng_offset
            ]
    
    # Save coordinates
    os.makedirs(os.path.dirname(coords_path), exist_ok=True)
    with open(coords_path, 'w') as f:
        json.dump(neighborhood_coords, f)
    
    return neighborhood_coords

# Load data and statistics
data = load_data()
stats = load_statistics()
neighborhood_coords = load_neighborhood_coords()

# Title
st.title("🏡 Zürich Real Estate Price Prediction")
st.write("Vorhersage von Immobilienpreisen in Zürich basierend auf Quartier, Baualter, Zimmeranzahl und Reisezeiten.")

# Sidebar for inputs
st.sidebar.header("Immobilien-Parameter")

# Get unique values from data or statistics
if len(data) > 0:
    quartiere = sorted(data['Quartier'].unique())
    baualter_list = sorted(data['Baualter'].unique())
    zimmer_list = sorted(data['Zimmeranzahl'].unique())
else:
    quartiere = stats.get('quartiere', ["City", "Oerlikon", "Altstetten", "Seefeld", "Enge"])
    baualter_list = stats.get('baualter', ["vor 1919", "1919-1945", "1946-1960", "1961-1980", "1981-2000", "nach 2000"])
    zimmer_list = stats.get('zimmeranzahlen', ["1 Zimmer", "2 Zimmer", "3 Zimmer", "4 Zimmer", "5+ Zimmer"])

# Sidebar inputs
selected_quartier = st.sidebar.selectbox("Quartier", quartiere)
selected_baualter = st.sidebar.selectbox("Baualter", baualter_list)
selected_zimmer = st.sidebar.selectbox("Zimmeranzahl", zimmer_list)

st.sidebar.subheader("Reisezeit-Präferenzen")
hauptbahnhof_limit = st.sidebar.slider("Max. Reisezeit zum Hauptbahnhof (Min)", 5, 60, 30)
eth_limit = st.sidebar.slider("Max. Reisezeit zur ETH Zürich (Min)", 5, 60, 30)
flughafen_limit = st.sidebar.slider("Max. Reisezeit zum Flughafen (Min)", 10, 90, 45)
bahnhofstrasse_limit = st.sidebar.slider("Max. Reisezeit zur Bahnhofstrasse (Min)", 5, 60, 30)

# Main content
tabs = st.tabs(["Preisvorhersage", "Interaktive Karte", "Datenanalyse"])

with tabs[0]:
    st.header("Immobilienpreis-Vorhersage")
    
    # Get or train model
    model = get_model(data)
    
    if model is None:
        st.error("Kein Modell verfügbar! Daten fehlen oder sind unzureichend.")
    else:
        # Make prediction
        st.subheader("Gewählte Parameter")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Quartier:** {selected_quartier}")
            st.write(f"**Baualter:** {selected_baualter}")
            st.write(f"**Zimmeranzahl:** {selected_zimmer}")
        
        with col2:
            st.write(f"**Reisezeit Hauptbahnhof:** max. {hauptbahnhof_limit} Min")
            st.write(f"**Reisezeit ETH:** max. {eth_limit} Min")
        
        with col3:
            st.write(f"**Reisezeit Flughafen:** max. {flughafen_limit} Min")
            st.write(f"**Reisezeit Bahnhofstrasse:** max. {bahnhofstrasse_limit} Min")
        
        # Create prediction data
        pred_data = {
            'Jahr': stats.get('neuestes_jahr', 2024),
            'Zimmer_Num': int(selected_zimmer.split()[0].replace('+', ''))
        }
        
        # Add travel times
        travel_times = {
            'Reisezeit_Hauptbahnhof': 30,
            'Reisezeit_ETH Zürich': 30,
            'Reisezeit_Flughafen Zürich': 45,
            'Reisezeit_Bahnhofstrasse': 30
        }
        
        # Load real travel times if available
        reisezeiten_path = 'reisezeiten.json'
        if os.path.exists(reisezeiten_path):
            with open(reisezeiten_path, 'r') as f:
                reisezeiten = json.load(f)
                if selected_quartier in reisezeiten:
                    quartier_zeiten = reisezeiten[selected_quartier]
                    travel_times = {
                        'Reisezeit_Hauptbahnhof': quartier_zeiten.get('Hauptbahnhof', 30),
                        'Reisezeit_ETH Zürich': quartier_zeiten.get('ETH Zürich', 30),
                        'Reisezeit_Flughafen Zürich': quartier_zeiten.get('Flughafen Zürich', 45),
                        'Reisezeit_Bahnhofstrasse': quartier_zeiten.get('Bahnhofstrasse', 30)
                    }
        
        # Add travel times to prediction data
        for key, value in travel_times.items():
            pred_data[key] = value
        
        # Create DataFrame for prediction
        X_pred = pd.DataFrame([pred_data])
        
        # Add one-hot encoded columns for neighborhoods and building age
        for col in model.feature_names_:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        # Set the selected neighborhood and building age
        quartier_col = f"Quartier_{selected_quartier}"
        if quartier_col in X_pred.columns:
            X_pred[quartier_col] = 1
        
        baualter_col = f"Baualter_{selected_baualter}"
        if baualter_col in X_pred.columns:
            X_pred[baualter_col] = 1
        
        # Ensure all required features are present
        missing_features = set(model.feature_names_) - set(X_pred.columns)
        if missing_features:
            for feature in missing_features:
                X_pred[feature] = 0
        
        # Ensure the order of features matches the model
        X_pred = X_pred[model.feature_names_]
        
        # Make prediction
        predicted_price = model.predict(X_pred)[0]
        
        # Display prediction
        st.subheader("Preisvorhersage")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Geschätzter Immobilienpreis", 
                value=f"CHF {predicted_price:,.0f}"
            )
        
        with col2:
            # Comparison with average prices
            if 'preis_avg' in stats:
                avg_price = stats['preis_avg']
                diff = (predicted_price - avg_price) / avg_price * 100
                st.metric(
                    label="Vergleich zum Durchschnitt", 
                    value=f"CHF {avg_price:,.0f}",
                    delta=f"{diff:.1f}%"
                )

        # Show travel times
        st.subheader("Reisezeiten vom gewählten Quartier")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hb_time = travel_times['Reisezeit_Hauptbahnhof']
            hb_color = "green" if hb_time <= hauptbahnhof_limit else "red"
            st.markdown(f"**Hauptbahnhof:** <span style='color:{hb_color}'>{hb_time} Min</span>", unsafe_allow_html=True)
        
        with col2:
            eth_time = travel_times['Reisezeit_ETH Zürich']
            eth_color = "green" if eth_time <= eth_limit else "red"
            st.markdown(f"**ETH Zürich:** <span style='color:{eth_color}'>{eth_time} Min</span>", unsafe_allow_html=True)
        
        with col3:
            flug_time = travel_times['Reisezeit_Flughafen Zürich']
            flug_color = "green" if flug_time <= flughafen_limit else "red"
            st.markdown(f"**Flughafen:** <span style='color:{flug_color}'>{flug_time} Min</span>", unsafe_allow_html=True)
        
        with col4:
            bahn_time = travel_times['Reisezeit_Bahnhofstrasse']
            bahn_color = "green" if bahn_time <= bahnhofstrasse_limit else "red"
            st.markdown(f"**Bahnhofstrasse:** <span style='color:{bahn_color}'>{bahn_time} Min</span>", unsafe_allow_html=True)

with tabs[1]:
    st.header("Immobilienpreise in Zürich")
    
    # Create a folium map
    m = folium.Map(location=ZURICH_COORDS, zoom_start=12)
    
    # Load travel times if available
    travel_times_available = False
    reisezeiten = {}
    
    reisezeiten_path = 'reisezeiten.json'
    if os.path.exists(reisezeiten_path):
        with open(reisezeiten_path, 'r') as f:
            reisezeiten = json.load(f)
            travel_times_available = True
    
    # Determine which travel time to display
    travel_time_options = ["Hauptbahnhof", "ETH Zürich", "Flughafen Zürich", "Bahnhofstrasse"]
    selected_dest = st.selectbox("Reisezeit anzeigen zu:", travel_time_options)
    
    # Get a subset of data for the map (latest year, if possible)
    if len(data) > 0:
        latest_year = data['Jahr'].max()
        map_data = data[data['Jahr'] == latest_year]
    else:
        map_data = pd.DataFrame()
    
    # Add markers for each neighborhood
    for neighborhood, coords in neighborhood_coords.items():
        # Get price data for this neighborhood if available
        if len(map_data) > 0:
            neighborhood_data = map_data[map_data['Quartier'] == neighborhood]
            if len(neighborhood_data) > 0:
                avg_price = neighborhood_data['GeschaetzterPreis'].mean()
                price_text = f"CHF {avg_price:,.0f}"
            else:
                price_text = "Keine Daten"
        else:
            price_text = "Keine Daten"
        
        # Get travel time if available
        travel_time = None
        if travel_times_available and neighborhood in reisezeiten:
            travel_time = reisezeiten[neighborhood].get(selected_dest, None)
        
        # Create popup text
        popup_text = f"""
        <b>{neighborhood}</b><br>
        Durchschnittspreis: {price_text}<br>
        """
        
        if travel_time is not None:
            popup_text += f"Reisezeit zum {selected_dest}: {travel_time} Min"
        
        # Determine marker color based on travel time
        if travel_time is not None:
            # Green if under 15min, yellow if under 30min, red otherwise
            if travel_time <= 15:
                color = 'green'
            elif travel_time <= 30:
                color = 'orange'
            else:
                color = 'red'
        else:
            color = 'blue'
        
        # Add marker
        folium.Marker(
            location=coords,
            popup=popup_text,
            tooltip=neighborhood,
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Show the map
    folium_static(m)
    
    # Legend
    st.markdown("""
    **Legende:**
    - 🟢 Reisezeit ≤ 15 Minuten
    - 🟠 Reisezeit ≤ 30 Minuten