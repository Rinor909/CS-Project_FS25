import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load CSV data from the specified path
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def preprocess_neighborhood_data(neighborhood_data):
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
    
    return data

def preprocess_building_age_data(building_age_data):
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
    
    return data

def preprocess_data(neighborhood_data, building_age_data):
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
    neighborhoods = sorted(neighborhood_df['neighborhood'].unique())
    room_counts = sorted(neighborhood_df['room_count'].unique())
    building_ages = sorted(building_age_df['building_age'].unique())
    
    # For now, we'll return the neighborhood data as our main dataset
    # In a real implementation, we would merge these datasets properly
    
    return neighborhood_df, neighborhoods, room_counts, building_ages, latest_year

def generate_travel_time(origin, destinations):
    """
    Generate travel time from origin to destinations
    This is a placeholder function that would normally use Google Maps API
    
    Args:
        origin: Origin location (neighborhood)
        destinations: List of destination locations
        
    Returns:
        dict: Mapping of destinations to travel times in minutes
    """
    # In a real implementation, this would call Google Maps API
    # For now, we'll generate random travel times
    import random
    
    travel_times = {}
    for dest in destinations:
        # Generate a random travel time between 5 and 45 minutes
        travel_times[dest] = random.randint(5, 45)
    
    return travel_times

def prepare_model_input(neighborhood, room_count, building_age, travel_times):
    """
    Prepare input features for the ML model
    
    Args:
        neighborhood: Selected neighborhood
        room_count: Selected room count
        building_age: Selected building age
        travel_times: Dictionary of travel times to key destinations
        
    Returns:
        DataFrame: Input features for the model
    """
    # In a real implementation, this would transform categorical variables
    # and create the proper feature set expected by the model
    
    # Create a dictionary of features
    features = {
        'neighborhood': [neighborhood],
        'room_count': [room_count],
        'building_age': [building_age]
    }
    
    # Add travel times
    for dest, time in travel_times.items():
        features[f'travel_time_{dest}'] = [time]
    
    return pd.DataFrame(features)