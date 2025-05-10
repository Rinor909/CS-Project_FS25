import pandas as pd
import numpy as np
import pickle
import os

def load_processed_data():
    """Loads the processed data for the app"""
    try:
        # Load data directly from GitHub URLs
        import requests
        from io import StringIO
        
        # Define GitHub URLs - make sure these are your actual URLs
        quartier_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/quartier_processed.csv"
        baualter_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/baualter_processed.csv"
        travel_times_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/travel_times.csv"
        
        print("Loading data from GitHub URLs...")
        
        # Load quartier data
        response = requests.get(quartier_url)
        if response.status_code == 200:
            df_quartier = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded quartier data: {len(df_quartier)} rows")
        else:
            print(f"Error loading quartier data: HTTP {response.status_code}")
            # Try loading from local file
            if os.path.exists('data/processed/quartier_processed.csv'):
                print("Trying to load quartier data from local file...")
                df_quartier = pd.read_csv('data/processed/quartier_processed.csv')
                print(f"Loaded quartier data from local file: {len(df_quartier)} rows")
            else:
                print("No quartier data found.")
                df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
                df_quartier['Jahr'] = [2024]  # Add default year
        
        # Load baualter data
        response = requests.get(baualter_url)
        if response.status_code == 200:
            df_baualter = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded baualter data: {len(df_baualter)} rows")
        else:
            print(f"Error loading baualter data: HTTP {response.status_code}")
            # Try loading from local file
            if os.path.exists('data/processed/baualter_processed.csv'):
                print("Trying to load baualter data from local file...")
                df_baualter = pd.read_csv('data/processed/baualter_processed.csv')
                print(f"Loaded baualter data from local file: {len(df_baualter)} rows")
            else:
                print("No baualter data found.")
                df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
        
        # Load travel times data
        response = requests.get(travel_times_url)
        if response.status_code == 200:
            df_travel_times = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded travel times data: {len(df_travel_times)} rows")
        else:
            print(f"Error loading travel times data: HTTP {response.status_code}")
            # Try loading from local file
            if os.path.exists('data/processed/travel_times.csv'):
                print("Trying to load travel times data from local file...")
                df_travel_times = pd.read_csv('data/processed/travel_times.csv')
                print(f"Loaded travel times data from local file: {len(df_travel_times)} rows")
            else:
                print("No travel times data found.")
                df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
        
        # Check for missing Jahr column and add if necessary
        if 'Jahr' not in df_quartier.columns:
            print("Adding missing Jahr column")
            df_quartier['Jahr'] = 2024
            
        return df_quartier, df_baualter, df_travel_times
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create empty DataFrames with expected columns as fallback
        df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
        df_quartier['Jahr'] = [2024]  # Add default year to avoid the Jahr error
        df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
        df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
        return df_quartier, df_baualter, df_travel_times

def load_model():
    """Loads the trained price prediction model from GitHub"""
    try:
        # Try to load model from GitHub
        import requests
        import io
        import pickle
        
        # GitHub URL to the model
        model_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/models/price_model.pkl"
        
        # Download model file
        print("Downloading model from GitHub...")
        response = requests.get(model_url)
        
        if response.status_code == 200:
            # Load the model from the response content
            model_data = io.BytesIO(response.content)
            model = pickle.load(model_data)
            print("Successfully loaded model from GitHub!")
            return model
        else:
            print(f"Failed to download model from GitHub: Status code {response.status_code}")
            
            # Try local file as fallback
            if os.path.exists('models/price_model.pkl'):
                print("Trying to load model from local file...")
                with open('models/price_model.pkl', 'rb') as file:
                    model = pickle.load(file)
                print("Successfully loaded model from local file!")
                return model
            
            print("Could not load model from GitHub or local file.")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_quartier_mapping():
    """Loads the neighborhood mapping data"""
    try:
        if not os.path.exists('models/quartier_mapping.pkl'):
            print("Warning: quartier_mapping.pkl not found")
            return {}
            
        with open('models/quartier_mapping.pkl', 'rb') as file:
            quartier_mapping = pickle.load(file)
        return quartier_mapping
    except Exception as e:
        print(f"Error loading quartier mapping: {e}")
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
        # Try to load from GitHub
        import requests
        from io import StringIO
        
        model_input_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv"
        
        print("Loading model input data from GitHub...")
        response = requests.get(model_input_url)
        
        if response.status_code == 200:
            df_final = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded model input data: {len(df_final)} rows")
        else:
            print(f"Error loading model input data: HTTP {response.status_code}")
            # Try local file
            if os.path.exists('data/processed/modell_input_final.csv'):
                print("Trying to load model input data from local file...")
                df_final = pd.read_csv('data/processed/modell_input_final.csv')
                print(f"Loaded model input data from local file: {len(df_final)} rows")
            else:
                print("No model input data found.")
                # Create a default DataFrame if file doesn't exist
                df_final = pd.DataFrame({
                    'Quartier_Code': [0],
                    'Quartier_Preisniveau': [1.0],
                    'MedianPreis_Baualter': [1000000]
                })
    except Exception as e:
        print(f"Error loading model input data: {e}")
        # Create a default DataFrame
        df_final = pd.DataFrame({
            'Quartier_Code': [0],
            'Quartier_Preisniveau': [1.0],
            'MedianPreis_Baualter': [1000000]
        })
    
    # Find average values for the selected neighborhood
    quartier_data = df_final[df_final['Quartier_Code'] == quartier_code] if 'Quartier_Code' in df_final.columns else pd.DataFrame()
    
    if len(quartier_data) > 0:
        print(f"Found data for neighborhood code {quartier_code}")
        quartier_preisniveau = quartier_data['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in quartier_data.columns else 1.0
        mediapreis_baualter = quartier_data['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in quartier_data.columns else 1000000
    else:
        print(f"No data found for neighborhood code {quartier_code}. Using average values.")
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
    
    print(f"Input data prepared: {input_data.columns.tolist()}")
    print(f"Input data sample: {input_data.iloc[0].to_dict()}")
    
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
        print("No model available for prediction.")
        
        # Only if absolutely necessary, use this as a last resort
        # This is needed to prevent the app from completely failing
        # when the model cannot be loaded for any reason
        quartier_preisniveau = input_data['Quartier_Preisniveau'].values[0] if 'Quartier_Preisniveau' in input_data.columns else 1.0
        mediapreis_baualter = input_data['MedianPreis_Baualter'].values[0] if 'MedianPreis_Baualter' in input_data.columns else 1000000
        zimmeranzahl = input_data['Zimmeranzahl_num'].values[0] if 'Zimmeranzahl_num' in input_data.columns else 3
        
        # Use the median price directly from input data and adjust by room count
        base_price = mediapreis_baualter
        room_factor = 1.0 + ((zimmeranzahl - 3) * 0.15)  # 15% per room difference from 3
        calculated_price = base_price * room_factor * quartier_preisniveau
        
        print(f"Using fallback calculation: {calculated_price:.2f} CHF")
        return calculated_price
    
    try:
        # Make sure we have all required columns for prediction
        print(f"Model features expected: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Unknown'}")
        print(f"Input data columns: {input_data.columns.tolist()}")
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        print(f"Model predicted: {prediction:.2f} CHF")
        
        # Check if prediction is reasonable
        if prediction <= 0 or prediction > 50000000:  # Sanity check
            print(f"Warning: Prediction outside reasonable range. Using median price.")
            return input_data['MedianPreis_Baualter'].values[0] if 'MedianPreis_Baualter' in input_data.columns else 1500000
            
        return round(prediction, 2)
    except Exception as e:
        print(f"Error in price prediction: {e}")
        
        # Use some values from input_data to create a more reasonable fallback
        try:
            quartier_preisniveau = input_data['Quartier_Preisniveau'].values[0] if 'Quartier_Preisniveau' in input_data.columns else 1.0
            mediapreis_baualter = input_data['MedianPreis_Baualter'].values[0] if 'MedianPreis_Baualter' in input_data.columns else 1000000
            zimmeranzahl = input_data['Zimmeranzahl_num'].values[0] if 'Zimmeranzahl_num' in input_data.columns else 3
            
            # Use the median price directly from input data
            calculated_price = mediapreis_baualter * quartier_preisniveau * (1.0 + ((zimmeranzahl - 3) * 0.15))
            print(f"Using calculated price: {calculated_price:.2f} CHF")
            return calculated_price
        except:
            # Absolute last resort
            return 1500000

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
        'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841}
    }