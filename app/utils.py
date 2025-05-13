import pandas as pd
import numpy as np
import pickle
import os
import requests
from io import StringIO

def load_processed_data():
    """Loads the processed data for the app"""
    # GitHub URLs for data sources
    quartier_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/quartier_processed.csv"
    baualter_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/baualter_processed.csv"
    travel_times_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/travel_times.csv"
    
    # Load quartier data
    try:
        df_quartier = pd.read_csv(quartier_url)
    except:
        # Fallback to local file or create empty DataFrame
        if os.path.exists('data/processed/quartier_processed.csv'):
            df_quartier = pd.read_csv('data/processed/quartier_processed.csv')
        else:
            df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
            df_quartier['Jahr'] = [2024]  # Add default year
    
    # Load baualter data
    try:
        df_baualter = pd.read_csv(baualter_url)
    except:
        # Fallback to local file or create empty DataFrame
        if os.path.exists('data/processed/baualter_processed.csv'):
            df_baualter = pd.read_csv('data/processed/baualter_processed.csv')
        else:
            df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
    
    # Load travel times data
    try:
        df_travel_times = pd.read_csv(travel_times_url)
    except:
        # Fallback to local file or create empty DataFrame
        if os.path.exists('data/processed/travel_times.csv'):
            df_travel_times = pd.read_csv('data/processed/travel_times.csv')
        else:
            df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
    
    # Ensure Jahr column exists
    if 'Jahr' not in df_quartier.columns:
        df_quartier['Jahr'] = 2024
        
    return df_quartier, df_baualter, df_travel_times

def load_model():
    """Loads the trained price prediction model"""
    model_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/models/price_model.pkl"
    
    try:
        # Try to load model from GitHub
        response = requests.get(model_url)
        if response.status_code == 200:
            model_data = StringIO(response.content.decode('latin1'))
            model = pickle.load(model_data)
            return model
        
        # Fallback to local file
        if os.path.exists('models/price_model.pkl'):
            with open('models/price_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
    except:
        pass
    
    # If model loading fails, return None (will use fallback calculation)
    return None

def load_quartier_mapping():
    """Loads the neighborhood mapping data"""
    try:
        if os.path.exists('models/quartier_mapping.pkl'):
            with open('models/quartier_mapping.pkl', 'rb') as file:
                return pickle.load(file)
    except:
        pass
    
    # Default empty mapping if loading fails
    return {}

def preprocess_input(quartier_code, zimmeranzahl, baujahr, travel_times_dict):
    """Prepares input data for the model"""
    # Load model input data
    try:
        model_input_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv"
        df_final = pd.read_csv(model_input_url)
    except:
        # Try local file
        try:
            if os.path.exists('data/processed/modell_input_final.csv'):
                df_final = pd.read_csv('data/processed/modell_input_final.csv')
            else:
                raise Exception()
        except:
            # Default values if loading fails
            df_final = pd.DataFrame({
                'Quartier_Code': [0],
                'Quartier_Preisniveau': [1.0],
                'MedianPreis_Baualter': [1000000]
            })
    
    # Find values for the selected neighborhood
    quartier_data = df_final[df_final['Quartier_Code'] == quartier_code] if 'Quartier_Code' in df_final.columns else pd.DataFrame()
    
    if len(quartier_data) > 0:
        quartier_preisniveau = quartier_data['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in quartier_data.columns else 1.0
        mediapreis_baualter = quartier_data['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in quartier_data.columns else 1000000
    else:
        # Fallback: Use average values or defaults
        quartier_preisniveau = df_final['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in df_final.columns else 1.0
        mediapreis_baualter = df_final['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in df_final.columns else 1000000
    
    # Create input data
    input_data = pd.DataFrame({
        'Quartier_Code': [quartier_code],
        'Zimmeranzahl_num': [zimmeranzahl],
        'PreisProQm': [quartier_preisniveau * 10000],  # Approximation
        'MedianPreis_Baualter': [mediapreis_baualter],
        'Durchschnitt_Baujahr': [baujahr],
        'Preis_Verhältnis': [1.0],  # Default value
        'Quartier_Preisniveau': [quartier_preisniveau]
    })
    
    # Add travel times
    for key, value in travel_times_dict.items():
        if key in ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']:
            input_data[f'Reisezeit_{key}'] = value
    
    return input_data

def predict_price(model, input_data):
    """Predicts price based on input data"""
    # Extract key values for fallback calculations
    quartier_code = input_data['Quartier_Code'].values[0] if 'Quartier_Code' in input_data.columns else 0
    zimmeranzahl = input_data['Zimmeranzahl_num'].values[0] if 'Zimmeranzahl_num' in input_data.columns else 3
    baujahr = input_data['Durchschnitt_Baujahr'].values[0] if 'Durchschnitt_Baujahr' in input_data.columns else 2000
    quartier_preisniveau = input_data['Quartier_Preisniveau'].values[0] if 'Quartier_Preisniveau' in input_data.columns else 1.0
    
    # Fallback calculation function
    def calculate_fallback_price():
        # Base price based on neighborhood price level
        base_price = 1000000 * quartier_preisniveau
        
        # Room factor: each room adds 15% to base price
        room_factor = 1.0 + ((zimmeranzahl - 3) * 0.15)
        
        # Age factor: newer buildings are more expensive
        aktuelles_jahr = 2025
        max_age = 100
        age = max(0, min(max_age, aktuelles_jahr - baujahr))
        age_factor = 1.3 - (age / max_age * 0.6)  # Ranges from 0.7 to 1.3
        
        return base_price * room_factor * age_factor
    
    # If no model, use fallback calculation
    if model is None:
        return calculate_fallback_price()
    
    try:
        # Make prediction using model
        prediction = model.predict(input_data)[0]
        
        # Apply sanity check - if prediction seems unreasonable, adjust based on construction year
        if prediction < 500000 or prediction > 10000000:
            aktuelles_jahr = 2025
            max_age = 100
            age = max(0, min(max_age, aktuelles_jahr - baujahr))
            age_factor = 1.3 - (age / max_age * 0.6)
            prediction = prediction * age_factor
        
        return round(prediction, 2)
    except:
        # If prediction fails, use fallback calculation
        return calculate_fallback_price()

def get_travel_times_for_quartier(quartier, df_travel_times, transportmittel='transit'):
    """Returns travel times for a specific neighborhood"""
    # Default travel times as fallback
    default_times = {
        'Hauptbahnhof': 20,
        'ETH': 25,
        'Flughafen': 35,
        'Bahnhofstrasse': 22
    }
    
    # Return defaults if data is missing or empty
    if df_travel_times.empty or 'Quartier' not in df_travel_times.columns:
        return default_times
    
    # Filter data for the selected neighborhood and transport mode
    filtered_data = df_travel_times[
        (df_travel_times['Quartier'] == quartier) & 
        (df_travel_times['Transportmittel'] == transportmittel)
    ]
    
    # Create travel times dictionary
    travel_times = {}
    for _, row in filtered_data.iterrows():
        travel_times[row['Zielort']] = row['Reisezeit_Minuten']
    
    # Return defaults if no matching data found
    return travel_times if travel_times else default_times

def get_quartier_statistics(quartier, df_quartier):
    """Calculates statistics for a specific neighborhood"""
    # Default statistics
    default_stats = {
        'median_preis': 1000000,
        'min_preis': 800000,
        'max_preis': 1200000,
        'preis_pro_qm': 10000,
        'anzahl_objekte': 0
    }
    
    # Return defaults if data is missing
    if 'Quartier' not in df_quartier.columns or df_quartier.empty:
        return default_stats
    
    # Filter data for selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # Return defaults if no matching data found
    if quartier_data.empty:
        return default_stats
    
    # Calculate statistics
    return {
        'median_preis': quartier_data['MedianPreis'].median() if 'MedianPreis' in quartier_data.columns else default_stats['median_preis'],
        'min_preis': quartier_data['MedianPreis'].min() if 'MedianPreis' in quartier_data.columns else default_stats['min_preis'],
        'max_preis': quartier_data['MedianPreis'].max() if 'MedianPreis' in quartier_data.columns else default_stats['max_preis'],
        'preis_pro_qm': quartier_data['PreisProQm'].median() if 'PreisProQm' in quartier_data.columns else default_stats['preis_pro_qm'],
        'anzahl_objekte': len(quartier_data)
    }

def get_price_history(quartier, df_quartier):
    """Returns price development for a specific neighborhood"""
    # Return empty DataFrame if data is missing
    if 'Quartier' not in df_quartier.columns or 'Jahr' not in df_quartier.columns or df_quartier.empty:
        return pd.DataFrame()
    
    # Filter data for selected neighborhood
    quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
    
    # Return empty DataFrame if no matching data found
    if quartier_data.empty:
        return pd.DataFrame()
    
    # Group by year and calculate median prices
    price_history = quartier_data.groupby('Jahr').agg({
        'MedianPreis': 'median',
        'PreisProQm': 'median' if 'PreisProQm' in quartier_data.columns else lambda x: None
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
    """Returns coordinates for all neighborhoods in Zurich"""
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