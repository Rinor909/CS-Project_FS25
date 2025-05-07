import pandas as pd
import numpy as np
import pickle
import os

def load_processed_data():
    """Loads the processed data for the app"""
    try:
        df_quartier = pd.read_csv('data/processed/quartier_processed.csv')
        df_baualter = pd.read_csv('data/processed/baualter_processed.csv')
        df_travel_times = pd.read_csv('data/processed/travel_times.csv')
        
        return df_quartier, df_baualter, df_travel_times
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        # Create empty DataFrames with expected columns as fallback
        df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
        df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
        df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
        return df_quartier, df_baualter, df_travel_times

def load_model():
    """Loads the trained price prediction model"""
    try:
        with open('models/price_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print("Model file not found. Please run model_training.py first.")
        return None

def load_quartier_mapping():
    """Loads the neighborhood mapping data"""
    try:
        with open('models/quartier_mapping.pkl', 'rb') as file:
            quartier_mapping = pickle.load(file)
        return quartier_mapping
    except FileNotFoundError:
        print("Quartier mapping file not found.")
        # Return a default empty mapping
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
    
    # Load neighborhood-specific statistics
    try:
        df_final = pd.read_csv('data/processed/modell_input_final.csv')
    except FileNotFoundError:
        # Create a default DataFrame if file doesn't exist
        df_final = pd.DataFrame({
            'Quartier_Code': [0],
            'Quartier_Preisniveau': [1.0],
            'MedianPreis_Baualter': [1000000]
        })
    
    # Find average values for the selected neighborhood
    quartier_data = df_final[df_final['Quartier_Code'] == quartier_code]
    
    if len(quartier_data) > 0:
        quartier_preisniveau = quartier_data['Quartier_Preisniveau'].mean()
        mediapreis_baualter = quartier_data['MedianPreis_Baualter'].mean()
    else:
        # Fallback: Use average values across all neighborhoods
        quartier_preisniveau = df_final['Quartier_Preisniveau'].mean()
        mediapreis_baualter = df_final['MedianPreis_Baualter'].mean()
    
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
        float: Predicted price
    """
    if model is None:
        # Return a default price if model is not available
        return 1000000
    
    try:
        prediction = model.predict(input_data)[0]
        return round(prediction, 2)
    except Exception as e:
        print(f"Error in price prediction: {e}")
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
        travel_times = {
            'Hauptbahnhof': 20,
            'ETH': 25,
            'Flughafen': 35,
            'Bahnhofstrasse': 22
        }
    
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
    # Filter data for the selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # If no data available, return default values
    if quartier_data.empty:
        return {
            'median_preis': 1000000,
            'min_preis': 800000,
            'max_preis': 1200000,
            'preis_pro_qm': 10000,
            'anzahl_objekte': 0
        }
    
    # Calculate statistics
    stats = {
        'median_preis': quartier_data['MedianPreis'].median(),
        'min_preis': quartier_data['MedianPreis'].min(),
        'max_preis': quartier_data['MedianPreis'].max(),
        'preis_pro_qm': quartier_data['PreisProQm'].median(),
        'anzahl_objekte': len(quartier_data)
    }
    
    return stats

def get_price_history(quartier, df_quartier):
    """
    Returns price development for a specific neighborhood
    
    Args:
        quartier (str): Neighborhood name
        df_quartier (pd.DataFrame): DataFrame with neighborhood data
        
    Returns:
        pd.DataFrame: Price development by year
    """
    # Filter data for the selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # If no data available, return empty DataFrame
    if quartier_data.empty:
        return pd.DataFrame()
    
    # Group by year and calculate average prices
    price_history = quartier_data.groupby('Jahr').agg({
        'MedianPreis': 'median',
        'PreisProQm': 'median'
    }).reset_index()
    
    return price_history

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
        'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841}
    }