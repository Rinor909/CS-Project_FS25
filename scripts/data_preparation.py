import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_processed_data_dir():
    """Create processed data directory if it doesn't exist"""
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir

def load_neighborhood_data():
    """Load and process the neighborhood dataset"""
    file_path = os.path.join('data', 'raw', 'bau515od5155.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading neighborhood data from {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Rename columns for clarity
    column_map = {
        'Stichtagdatjahr': 'year',
        'RaumLang': 'neighborhood',
        'AnzZimmerLevel2Lang_noDM': 'room_count',
        'HAMedianPreis': 'median_price',
        'HAPreisWohnflaeche': 'price_per_sqm',
        'HAArtLevel1Lang': 'property_type'
    }
    
    # Select necessary columns
    selected_cols = list(column_map.keys())
    df = df[selected_cols].rename(columns=column_map)
    
    # Filter for apartments only
    df = df[df['property_type'] == 'Wohnungen']
    
    # Remove missing values
    df = df.dropna(subset=['median_price', 'neighborhood', 'room_count'])
    
    # Convert price columns to numeric
    df['median_price'] = pd.to_numeric(df['median_price'], errors='coerce')
    df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')
    
    print(f"Processed neighborhood data: {len(df)} rows")
    
    return df

def load_building_age_data():
    """Load and process the building age dataset"""
    file_path = os.path.join('data', 'raw', 'bau515od5156.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading building age data from {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Rename columns for clarity
    column_map = {
        'Stichtagdatjahr': 'year',
        'BaualterLang_noDM': 'building_age',
        'AnzZimmerLevel2Lang_noDM': 'room_count',
        'HAMedianPreis': 'median_price',
        'HAPreisWohnflaeche': 'price_per_sqm',
        'HAArtLevel1Lang': 'property_type'
    }
    
    # Select necessary columns
    selected_cols = list(column_map.keys())
    df = df[selected_cols].rename(columns=column_map)
    
    # Filter for apartments only
    df = df[df['property_type'] == 'Wohnungen']
    
    # Remove missing values
    df = df.dropna(subset=['median_price', 'building_age', 'room_count'])
    
    # Convert price columns to numeric
    df['median_price'] = pd.to_numeric(df['median_price'], errors='coerce')
    df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')
    
    print(f"Processed building age data: {len(df)} rows")
    
    return df

def save_processed_data(neighborhood_df, building_age_df, processed_dir):
    """Save processed data to CSV files"""
    # Save neighborhood data
    neighborhood_path = os.path.join(processed_dir, 'processed_neighborhood_data.csv')
    neighborhood_df.to_csv(neighborhood_path, index=False)
    print(f"Saved processed neighborhood data to {neighborhood_path}")
    
    # Save building age data
    building_age_path = os.path.join(processed_dir, 'processed_building_age_data.csv')
    building_age_df.to_csv(building_age_path, index=False)
    print(f"Saved processed building age data to {building_age_path}")
    
    # Create a combined dataset for the latest year
    latest_year = max(neighborhood_df['year'].max(), building_age_df['year'].max())
    
    latest_neighborhood = neighborhood_df[neighborhood_df['year'] == latest_year]
    latest_building_age = building_age_df[building_age_df['year'] == latest_year]
    
    # Save latest year data
    latest_neighborhood_path = os.path.join(processed_dir, f'latest_{latest_year}_neighborhood_data.csv')
    latest_neighborhood.to_csv(latest_neighborhood_path, index=False)
    print(f"Saved latest neighborhood data to {latest_neighborhood_path}")
    
    latest_building_age_path = os.path.join(processed_dir, f'latest_{latest_year}_building_age_data.csv')
    latest_building_age.to_csv(latest_building_age_path, index=False)
    print(f"Saved latest building age data to {latest_building_age_path}")

def main():
    """Main function to process data"""
    print("Starting data preparation...")
    
    # Create processed data directory
    processed_dir = create_processed_data_dir()
    
    try:
        # Load and process data
        neighborhood_df = load_neighborhood_data()
        building_age_df = load_building_age_data()
        
        # Save processed data
        save_processed_data(neighborhood_df, building_age_df, processed_dir)
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()