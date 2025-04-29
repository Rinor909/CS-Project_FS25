"""
Utility Functions for Zurich Real Estate Price Prediction App
------------------------------------------------------------
Purpose: Helper functions for data loading, processing, and model prediction

Tasks:
1. Load trained model
2. Load and process data for visualization
3. Make price predictions
4. Format results for display

Owner: Matteo (Primary), Rinor (Support)
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
MODELS_DIR = "../models"
PROCESSED_DATA_DIR = "../data/processed"
MODEL_FILE = "price_model.pkl"
NEIGHBORHOOD_DATA = "processed_neighborhood.csv"
BUILDING_AGE_DATA = "processed_building_age.csv"
TRAVEL_TIME_DATA = "neighborhood_travel_times.csv"

def load_model():
    """Load trained model from disk."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        features = model_data['features']
        
        logger.info(f"Model loaded from {model_path}")
        return model, features
    
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def load_datasets():
    """Load processed datasets for visualization."""
    try:
        # Load neighborhood data
        neighborhood_path = os.path.join(PROCESSED_DATA_DIR, NEIGHBORHOOD_DATA)
        neighborhood_df = pd.read_csv(neighborhood_path)
        
        # Load building age data
        building_age_path = os.path.join(PROCESSED_DATA_DIR, BUILDING_AGE_DATA)
        building_age_df = pd.read_csv(building_age_path)
        
        # Load travel time data
        travel_time_path = os.path.join(PROCESSED_DATA_DIR, TRAVEL_TIME_DATA)
        travel_time_df = pd.read_csv(travel_time_path)
        
        logger.info("Datasets loaded successfully")
        return neighborhood_df, building_age_df, travel_time_df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return None, None, None

def prepare_prediction_input(neighborhood, room_count, building_age, travel_times):
    """
    Prepare input data for model prediction.
    
    Parameters:
    - neighborhood: str, neighborhood name
    - room_count: int, number of rooms
    - building_age: int, age of building in years
    - travel_times: dict, travel times to key destinations
    
    Returns:
    - DataFrame with features ready for prediction
    """
    # This is a placeholder - implement actual feature preparation
    # based on your model's requirements
    
    # Example:
    data = {
        'neighborhood_factor': [0.5],  # This should be derived from actual neighborhood
        'building_age': [building_age],
        'room_count': [room_count],
        'travel_time_hauptbahnhof': [travel_times.get('Hauptbahnhof', 20)],
        'travel_time_eth': [travel_times.get('ETH_Zurich', 25)],
        'travel_time_airport': [travel_times.get('Zurich_Airport', 40)],
        'travel_time_bahnhofstrasse': [travel_times.get('Bahnhofstrasse', 20)]
    }
    
    return pd.DataFrame(data)

def predict_price(model, features, input_data):
    """
    Make price prediction using trained model.
    
    Parameters:
    - model: trained model object
    - features: list of feature names expected by the model
    - input_data: DataFrame with input features
    
    Returns:
    - Predicted price
    """
    if model is None:
        logger.error("No model available for prediction")
        return None
    
    try:
        # Ensure input data has all required features
        for feature in features:
            if feature not in input_data.columns:
                logger.warning(f"Missing feature: {feature}")
                input_data[feature] = 0  # Default value
        
        # Keep only the features used by the model
        input_data = input_data[features]
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        logger.info(f"Price prediction: {predicted_price:.2f}")
        return predicted_price
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def format_price(price):
    """Format price for display."""
    if price is None:
        return "N/A"
    
    return f"CHF {price:,.0f}"

def get_price_range(price):
    """Get price range (±10%)."""
    if price is None:
        return "N/A", "N/A"
    
    lower_bound = price * 0.9
    upper_bound = price * 1.1
    
    return f"CHF {lower_bound:,.0f}", f"CHF {upper_bound:,.0f}"

def get_neighborhoods():
    """Get list of neighborhoods in Zurich."""
    # This is a placeholder - replace with actual neighborhood data
    neighborhoods = [
        "Kreis 1",
        "Kreis 2",
        "Kreis 3",
        "Kreis 4",
        "Kreis 5",
        "Kreis 6",
        "Kreis 7",
        "Kreis 8",
        "Kreis 9",
        "Kreis 10",
        "Kreis 11",
        "Kreis 12"
    ]
    
    return neighborhoods

def get_neighborhood_coordinates():
    """
    Get coordinates for Zurich neighborhoods.
    
    Returns:
    - Dictionary mapping neighborhood names to coordinates
    """
    # This is a placeholder - replace with actual coordinate data
    coordinates = {
        "Kreis 1": {"lat": 47.3723, "lng": 8.5398},
        "Kreis 2": {"lat": 47.3605, "lng": 8.5244},
        "Kreis 3": {"lat": 47.3708, "lng": 8.5018},
        "Kreis 4": {"lat": 47.3792, "lng": 8.5198},
        "Kreis 5": {"lat": 47.3887, "lng": 8.5293},
        "Kreis 6": {"lat": 47.3899, "lng": 8.5500},
        "Kreis 7": {"lat": 47.3663, "lng": 8.5685},
        "Kreis 8": {"lat": 47.3502, "lng": 8.5685},
        "Kreis 9": {"lat": 47.3870, "lng": 8.4903},
        "Kreis 10": {"lat": 47.4104, "lng": 8.5090},
        "Kreis 11": {"lat": 47.4137, "lng": 8.5425},
        "Kreis 12": {"lat": 47.3950, "lng": 8.5698}
    }
    
    return coordinates

def get_key_destinations():
    """Get coordinates for key destinations in Zurich."""
    destinations = {
        "Hauptbahnhof": {"lat": 47.3782, "lng": 8.5401},
        "ETH_Zurich": {"lat": 47.3763, "lng": 8.5475},
        "Zurich_Airport": {"lat": 47.4502, "lng": 8.5614},
        "Bahnhofstrasse": {"lat": 47.3723, "lng": 8.5390}
    }
    
    return destinations

def get_building_age_options():
    """Get building age categories."""
    # This is a placeholder - replace with actual age categories
    age_categories = [
        "Before 1919",
        "1919-1945",
        "1946-1960",
        "1961-1970",
        "1971-1980",
        "1981-1990",
        "1991-2000",
        "2001-2010",
        "After 2010"
    ]
    
    return age_categories

def get_room_count_options():
    """Get room count options."""
    return [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

def convert_age_category_to_years(age_category):
    """Convert age category to approximate building age in years."""
    current_year = datetime.now().year
    
    age_mapping = {
        "Before 1919": current_year - 1919,
        "1919-1945": current_year - 1932,  # Midpoint
        "1946-1960": current_year - 1953,
        "1961-1970": current_year - 1965,
        "1971-1980": current_year - 1975,
        "1981-1990": current_year - 1985,
        "1991-2000": current_year - 1995,
        "2001-2010": current_year - 2005,
        "After 2010": current_year - 2015
    }
    
    return age_mapping.get(age_category, 50)  # Default to 50 years if not found
