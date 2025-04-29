import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Tuple, List, Dict, Optional, Any, Union

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from the specified path
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def preprocess_neighborhood_data(neighborhood_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the neighborhood dataset (bau515od5155.csv)
    
    Args:
        neighborhood_data: Raw neighborhood data
        
    Returns:
        DataFrame: Processed neighborhood data
    """
    # Rename columns for clarity
    data = neighborhood_data.copy()
    
    # Create a mapping of columns
    column_map = {
        'Stichtagdatjahr': 'year',
        'RaumLang': 'neighborhood',
        'AnzZimmerLevel2Lang_noDM': 'room_count',
        'HAMedianPreis': 'median_price',
        'HAPreisWohnflaeche': 'price_per_sqm'
    }
    
    # Select and rename columns
    data = data[list(column_map.keys())].rename(columns=column_map)
    
    # Filter for only properties (Wohnungen)
    if 'HAArtLevel1Lang' in neighborhood_data.columns:
        data = neighborhood_data[neighborhood_data['HAArtLevel1Lang'] == 'Wohnungen']
        # Add the renamed columns after filtering
        data = data[list(column_map.keys())].rename(columns=column_map)
    
    # Convert price columns to numeric
    data['median_price'] = pd.to_numeric(data['median_price'], errors='coerce')
    data['price_per_sqm'] = pd.to_numeric(data['price_per_sqm'], errors='coerce')
    
    # Drop rows with missing prices
    data = data.dropna(subset=['median_price'])
    
    return data

def preprocess_building_age_data(building_age_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the building age dataset (bau515od5156.csv)
    
    Args:
        building_age_data: Raw building age data
        
    Returns:
        DataFrame: Processed building age data
    """
    # Rename columns for clarity
    data = building_age_data.copy()
    
    # Create a mapping of columns
    column_map = {
        'Stichtagdatjahr': 'year',
        'BaualterLang_noDM': 'building_age',
        'AnzZimmerLevel2Lang_noDM': 'room_count',
        'HAMedianPreis': 'median_price',
        'HAPreisWohnflaeche': 'price_per_sqm'
    }
    
    # Select and rename columns
    data = data[list(column_map.keys())].rename(columns=column_map)
    
    # Filter for only properties (Wohnungen)
    if 'HAArtLevel1Lang' in building_age_data.columns:
        data = building_age_data[building_age_data['HAArtLevel1Lang'] == 'Wohnungen']
        # Add the renamed columns after filtering
        data = data[list(column_map.keys())].rename(columns=column_map)
    
    # Convert price columns to numeric
    data['median_price'] = pd.to_numeric(data['median_price'], errors='coerce')
    data['price_per_sqm'] = pd.to_numeric(data['price_per_sqm'], errors='coerce')
    
    # Drop rows with missing prices
    data = data.dropna(subset=['median_price'])
    
    return data

def preprocess_data(
    neighborhood_data: pd.DataFrame, 
    building_age_data: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], int]:
    """
    Preprocess both datasets and prepare for analysis
    
    Args:
        neighborhood_data: Raw neighborhood data
        building_age_data: Raw building age data
        
    Returns:
        tuple: (combined_data, neighborhoods_list, room_counts_list, building_ages_list, latest_year)
    """
    # Process each dataset
    neighborhood_df = preprocess_neighborhood_data(neighborhood_data)
    building_age_df = preprocess_building_age_data(building_age_data)
    
    # Get the latest year in the data
    latest_year = max(neighborhood_df['year'].max(), building_age_df['year'].max())
    
    # Get unique neighborhoods, room counts, and building ages
    neighborhoods = sorted(neighborhood_df['neighborhood'].unique().tolist())
    room_counts = sorted(neighborhood_df['room_count'].unique().tolist())
    building_ages = sorted(building_age_df['building_age'].unique().tolist())
    
    # For now, we'll return the neighborhood data as our main dataset
    # In a real implementation, we would merge these datasets properly
    
    return neighborhood_df, neighborhoods, room_counts, building_ages, latest_year

def load_travel_times(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load travel time data from JSON file
    
    Args:
        file_path: Path to the travel times JSON file
        
    Returns:
        dict: Nested dictionary of travel times by neighborhood and destination
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Travel times file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def load_model(model_path: str) -> Any:
    """
    Load the trained ML model
    
    Args:
        model_path: Path to the pickled model file
        
    Returns:
        object: Trained model
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def format_price(price: float) -> str:
    """
    Format price with thousands separator and CHF
    
    Args:
        price: Price value
        
    Returns:
        str: Formatted price string
    """
    return f"{price:,.0f} CHF"

def calculate_price_per_room(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price per room for the dataset
    
    Args:
        df: DataFrame with median_price and room_count columns
        
    Returns:
        DataFrame: Original DataFrame with price_per_room column added
    """
    result = df.copy()
    
    # Extract room count numerically (handle ranges like "3-4" rooms)
    def extract_room_number(room_str):
        if isinstance(room_str, str):
            if '-' in room_str:
                # For ranges like "3-4", take the average
                parts = room_str.split('-')
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return np.nan
            else:
                # Try to extract a number
                import re
                numbers = re.findall(r'\d+', room_str)
                if numbers:
                    return float(numbers[0])
        return np.nan
    
    # Add room count numeric column
    result['room_count_numeric'] = result['room_count'].apply(extract_room_number)
    
    # Calculate price per room
    result['price_per_room'] = result['median_price'] / result['room_count_numeric']
    
    return result
