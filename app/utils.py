import pandas as pd
import numpy as np
import pickle
import os
import requests
from io import StringIO
import streamlit as st

# GitHub raw data URLs - Replace with your actual GitHub username and repository
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main"
QUARTIER_PROCESSED_URL = f"{GITHUB_RAW_BASE}/data/processed/quartier_processed.csv"
BAUALTER_PROCESSED_URL = f"{GITHUB_RAW_BASE}/data/processed/baualter_processed.csv"
TRAVEL_TIMES_URL = f"{GITHUB_RAW_BASE}/data/processed/travel_times.csv"
MODEL_INPUT_URL = f"{GITHUB_RAW_BASE}/data/processed/modell_input_final.csv"
MODEL_URL = f"{GITHUB_RAW_BASE}/models/price_model.pkl"
QUARTIER_MAPPING_URL = f"{GITHUB_RAW_BASE}/models/quartier_mapping.pkl"

def load_processed_data():
    """Loads the processed data from GitHub URLs"""
    try:
        # Try to download quartier data
        print("Loading quartier data from GitHub...")
        response = requests.get(QUARTIER_PROCESSED_URL)
        if response.status_code == 200:
            df_quartier = pd.read_csv(StringIO(response.text))
            print(f"Loaded {len(df_quartier)} quartier records.")
        else:
            print(f"Failed to load quartier data: HTTP {response.status_code}")
            df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
        
        # Try to download baualter data
        print("Loading baualter data from GitHub...")
        response = requests.get(BAUALTER_PROCESSED_URL)
        if response.status_code == 200:
            df_baualter = pd.read_csv(StringIO(response.text))
            print(f"Loaded {len(df_baualter)} baualter records.")
        else:
            print(f"Failed to load baualter data: HTTP {response.status_code}")
            df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
        
        # Try to download travel times data
        print("Loading travel times data from GitHub...")
        response = requests.get(TRAVEL_TIMES_URL)
        if response.status_code == 200:
            df_travel_times = pd.read_csv(StringIO(response.text))
            print(f"Loaded {len(df_travel_times)} travel time records.")
        else:
            print(f"Failed to load travel times data: HTTP {response.status_code}")
            df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
        
        return df_quartier, df_baualter, df_travel_times
    except Exception as e:
        print(f"Error loading data from GitHub: {e}")
        st.error(f"Error loading data: {e}")
        # Create empty DataFrames with expected columns as fallback
        df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
        df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
        df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
        return df_quartier, df_baualter, df_travel_times

def load_model():
    """Loads the trained price prediction model from GitHub"""
    try:
        # First try with pickle - preferred method
        print("Loading model from GitHub using pickle...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            # Load the model from the response content
            import io
            model = pickle.load(io.BytesIO(response.content))
            if hasattr(model, 'predict'):
                print("Successfully loaded model with pickle.")
                return model
            else:
                print("Loaded object is not a valid model (no predict method).")
        else:
            print(f"Failed to load model pickle: HTTP {response.status_code}")
            
        # Create a simple model as fallback
        print("Creating a fallback model...")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Get training data from GitHub
        df_input = pd.read_csv(MODEL_INPUT_URL)
        if len(df_input) > 0:
            # Basic features for a simple model
            X = df_input[['Quartier_Code', 'Zimmeranzahl_num', 'Quartier_Preisniveau']].fillna(0)
            y = df_input['MedianPreis']
            model.fit(X, y)
            print("Trained a fallback model on GitHub data.")
            return model
        else:
            print("No training data available for fallback model.")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

def load_quartier_mapping():
    """Loads the neighborhood mapping data from GitHub"""
    try:
        # First try to use pickle
        print("Loading quartier mapping from GitHub...")
        response = requests.get(QUARTIER_MAPPING_URL)
        if response.status_code == 200:
            # Load the mapping from the response content
            import io
            quartier_mapping = pickle.load(io.BytesIO(response.content))
            print(f"Loaded mapping for {len(quartier_mapping)} neighborhoods.")
            return quartier_mapping
        else:
            print(f"Failed to load quartier mapping: HTTP {response.status_code}")
        
        # Create a fallback mapping from the input data
        print("Creating fallback quartier mapping...")
        response = requests.get(MODEL_INPUT_URL)
        if response.status_code == 200:
            df_input = pd.read_csv(StringIO(response.text))
            quartier_mapping = {code: quartier for code, quartier in zip(df_input['Quartier_Code'], df_input['Quartier'])}
            print(f"Created fallback mapping for {len(quartier_mapping)} neighborhoods.")
            return quartier_mapping
        else:
            print("Could not create fallback quartier mapping.")
            return {}
            
    except Exception as e:
        print(f"Error loading quartier mapping: {e}")
        st.error(f"Error loading quartier mapping: {e}")
        return {}

def preprocess_input(quartier_code, zimmeranzahl, baujahr, travel_times_dict):
    """
    Prepares input data for the model
    
    Args:
        quartier_code (int): Neighborhood code
        zimmeranzahl (int): Number of rooms
        baujahr (int): Construction year
        travel_times_dict (dict): Travel times to important locations
        
    Returns:
        pd.DataFrame: Preprocessed data for prediction
    """
    # Calculate average age in years
    aktuelles_jahr = 2025  # Current year
    alter = aktuelles_jahr - baujahr
    
    # Load neighborhood-specific statistics from GitHub
    try:
        print("Loading model input data from GitHub...")
        response = requests.get(MODEL_INPUT_URL)
        if response.status_code == 200:
            df_final = pd.read_csv(StringIO(response.text))
            print(f"Loaded {len(df_final)} model input records.")
        else:
            print(f"Failed to load model input data: HTTP {response.status_code}")
            # Create a default DataFrame if file doesn't exist
            df_final = pd.DataFrame({
                'Quartier_Code': [0],
                'Quartier_Preisniveau': [1.0],
                'MedianPreis_Baualter': [1000000]
            })
    except Exception as e:
        print(f"Error loading model input data from GitHub: {e}")
        # Create a default DataFrame
        df_final = pd.DataFrame({
            'Quartier_Code': [0],
            'Quartier_Preisniveau': [1.0],
            'MedianPreis_Baualter': [1000000]
        })
    
    # Find average values for the selected neighborhood
    quartier_data = df_final[df_final['Quartier_Code'] == quartier_code] if 'Quartier_Code' in df_final.columns else pd.DataFrame()
    
    if len(quartier_data) > 0:
        quartier_preisniveau = quartier_data['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in quartier_data.columns else 1.0
        mediapreis_baualter = quartier_data['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in quartier_data.columns else 1000000
    else:
        # Fallback: Use average values across all neighborhoods
        quartier_preisniveau = df_final['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in df_final.columns else 1.0
        mediapreis_baualter = df_final['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in df_final.columns else 1000000
    
    # Create input data as DataFrame
    input_data = pd.DataFrame({
        'Quartier_Code': [quartier_code],
        'Zimmeranzahl_num': [zimmeranzahl],
        'PreisProQm': [quartier_preisniveau * 10000],  # Approximation
        'MedianPreis_Baualter': [mediapreis_baualter],
        'Durchschnitt_Baujahr': [baujahr],
        'Preis_Verhältnis': [1.0],  # Default value, will be adjusted by the model
        'Quartier_Preisniveau': [quartier_preisniveau]
    })
    
    # Add travel times if available
    for key, value in travel_times_dict.items():
        if key in ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']:
            input_data[f'Reisezeit_{key}'] = value
    
    return input_data

def predict_price(model, input_data):
    """
    Predicts price based on input data
    
    Args:
        model: Trained ML model
        input_data (pd.DataFrame): Preprocessed input data
        
    Returns:
        float: Predicted price or None if prediction fails
    """
    if model is None:
        print("No valid model available for prediction.")
        return None
    
    try:
        # Ensure the input data has the expected format
        if len(input_data) == 0:
            print("Empty input data for prediction.")
            return None
            
        # Try to make a prediction
        prediction = model.predict(input_data)[0]
        
        # Check if the prediction is reasonable
        if prediction <= 0 or prediction > 50000000:  # Sanity check
            print(f"Prediction outside reasonable range: {prediction}")
            return None
            
        return round(prediction, 2)
    except Exception as e:
        print(f"Error in price prediction: {e}")
        st.error(f"Error making prediction: {e}")
        return None

def get_travel_times_for_quartier(quartier, df_travel_times, transportmittel='transit'):
    """
    Returns travel times for a specific neighborhood
    
    Args:
        quartier (str): Neighborhood name
        df_travel_times (pd.DataFrame): DataFrame with travel time data
        transportmittel (str): 'transit' or 'driving'
        
    Returns:
        dict: Travel times to different destinations
    """
    # Default travel times as fallback
    default_times = {
        'Hauptbahnhof': 20,
        'ETH': 25,
        'Flughafen': 35,
        'Bahnhofstrasse': 22
    }
    
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten']
    if any(col not in df_travel_times.columns for col in required_columns) or df_travel_times.empty:
        return default_times
    
    # Filter data for the selected neighborhood and transport mode
    filtered_data = df_travel_times[
        (df_travel_times['Quartier'] == quartier) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ]
    
    # Group travel times by destination
    travel_times = {}
    for _, row in filtered_data.iterrows():
        travel_times[row['Zielort']] = row['Reisezeit_Minuten']
    
    # If no data available, return default values
    if not travel_times:
        return default_times
    
    return travel_times

def get_quartier_statistics(quartier, df_quartier):
    """
    Calculates statistics for a specific neighborhood
    
    Args:
        quartier (str): Neighborhood name
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        
    Returns:
        dict: Statistics for the neighborhood
    """
    # Default statistics as fallback
    default_stats = {
        'median_preis': 1000000,
        'min_preis': 800000,
        'max_preis': 1200000,
        'preis_pro_qm': 10000,
        'anzahl_objekte': 0
    }
    
    # Check if necessary columns exist
    if 'Quartier' not in df_quartier.columns or df_quartier.empty:
        return default_stats
    
    # Filter data for the selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # If no data available, return default values
    if quartier_data.empty:
        return default_stats
    
    # Calculate statistics
    try:
        stats = {
            'median_preis': quartier_data['MedianPreis'].median() if 'MedianPreis' in quartier_data.columns else default_stats['median_preis'],
            'min_preis': quartier_data['MedianPreis'].min() if 'MedianPreis' in quartier_data.columns else default_stats['min_preis'],
            'max_preis': quartier_data['MedianPreis'].max() if 'MedianPreis' in quartier_data.columns else default_stats['max_preis'],
            'preis_pro_qm': quartier_data['PreisProQm'].median() if 'PreisProQm' in quartier_data.columns else default_stats['preis_pro_qm'],
            'anzahl_objekte': len(quartier_data)
        }
        return stats
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return default_stats

def get_price_history(quartier, df_quartier):
    """
    Returns price development for a specific neighborhood
    
    Args:
        quartier (str): Neighborhood name
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        
    Returns:
        pd.DataFrame: Price development by year
    """
    # Check if necessary columns exist
    required_columns = ['Quartier', 'Jahr', 'MedianPreis']
    if any(col not in df_quartier.columns for col in required_columns) or df_quartier.empty:
        return pd.DataFrame()
    
    # Filter data for the selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # If no data available, return empty DataFrame
    if quartier_data.empty:
        return pd.DataFrame()
    
    try:
        # Group by year and calculate average prices
        price_history = quartier_data.groupby('Jahr').agg({
            'MedianPreis': 'median',
            'PreisProQm': 'median' if 'PreisProQm' in quartier_data.columns else lambda x: None
        }).reset_index()
        
        return price_history
    except Exception as e:
        print(f"Error creating price history: {e}")
        return pd.DataFrame()

def get_zurich_coordinates():
    """Returns coordinates for Zurich"""
    return {
        'latitude': 47.3769,
        'longitude': 8.5417,
        'zoom': 12
    }

def get_quartier_coordinates():
    """
    Returns coordinates for all neighborhoods in Zurich
    In a real application, these would be loaded from a geodatabase or GeoJSON
    """
    # These values are approximations - in a real application, 
    # the exact polygons or centroids of the neighborhoods would be used
    return {
        'Hottingen': {'lat': 47.3692, 'lng': 8.5631},
        'Fluntern': {'lat': 47.3809, 'lng': 8.5629},
        'Unterstrass': {'lat': 47.3864, 'lng': 8.5419},
        'Oberstrass': {'lat': 47.3889, 'lng': 8.5481},
        'Rathaus': {'lat': 47.3716, 'lng': 8.5428},
        'Lindenhof': {'lat': 47.3728, 'lng': 8.5408},
        'City': {'lat': 47.3752, 'lng': 8.5385},
        'Seefeld': {'lat': 47.3600, 'lng': 8.5532},
        'Mühlebach': {'lat': 47.3638, 'lng': 8.5471},
        'Witikon': {'lat': 47.3610, 'lng': 8.5881},
        'Hirslanden': {'lat': 47.3624, 'lng': 8.5705},
        'Enge': {'lat': 47.3628, 'lng': 8.5288},
        'Wollishofen': {'lat': 47.3489, 'lng': 8.5266},
        'Leimbach': {'lat': 47.3279, 'lng': 8.5098},
        'Friesenberg': {'lat': 47.3488, 'lng': 8.5035},
        'Alt-Wiedikon': {'lat': 47.3652, 'lng': 8.5158},
        'Sihlfeld': {'lat': 47.3742, 'lng': 8.5072},
        'Albisrieden': {'lat': 47.3776, 'lng': 8.4842},
        'Altstetten': {'lat': 47.3917, 'lng': 8.4876},
        'Höngg': {'lat': 47.4023, 'lng': 8.4976},
        'Wipkingen': {'lat': 47.3930, 'lng': 8.5253},
        'Affoltern': {'lat': 47.4230, 'lng': 8.5047},
        'Oerlikon': {'lat': 47.4126, 'lng': 8.5487},
        'Seebach': {'lat': 47.4258, 'lng': 8.5422},
        'Saatlen': {'lat': 47.4087, 'lng': 8.5742},
        'Schwamendingen-Mitte': {'lat': 47.4064, 'lng': 8.5648},
        'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841},
        'Ganze Stadt': {'lat': 47.3769, 'lng': 8.5417},
        'Kreis 1': {'lat': 47.3732, 'lng': 8.5413},
        'Kreis 2': {'lat': 47.3559, 'lng': 8.5277},
        'Kreis 3': {'lat': 47.3682, 'lng': 8.5097},
        'Kreis 4': {'lat': 47.3767, 'lng': 8.5257},
        'Kreis 5': {'lat': 47.3875, 'lng': 8.5295},
        'Kreis 6': {'lat': 47.3846, 'lng': 8.5498},
        'Kreis 7': {'lat': 47.3637, 'lng': 8.5751},
        'Kreis 8': {'lat': 47.3560, 'lng': 8.5513},
        'Kreis 9': {'lat': 47.3796, 'lng': 8.4882},
        'Kreis 10': {'lat': 47.4088, 'lng': 8.5253},
        'Kreis 11': {'lat': 47.4173, 'lng': 8.5456},
        'Kreis 12': {'lat': 47.3985, 'lng': 8.5761},
        'Hochschulen': {'lat': 47.3743, 'lng': 8.5482},
        'Langstrasse': {'lat': 47.3800, 'lng': 8.5300},
        'Wurde-Furrer': {'lat': 47.3791, 'lng': 8.5261},
        'Escher-Wyss': {'lat': 47.3888, 'lng': 8.5219},
        'Gewerbeschule': {'lat': 47.3850, 'lng': 8.5311},
        'Hard': {'lat': 47.3832, 'lng': 8.5198}
    }