"""
Data Preparation Script for Zurich Real Estate Price Prediction
----------------------------------------------------------------
Purpose: Clean and preprocess raw datasets for model training

Tasks:
1. Load raw datasets (neighborhood and building age)
2. Clean missing values and handle outliers
3. Merge datasets if necessary
4. Create derived features
5. Export processed datasets to ../data/processed/

Owner: Rinor (Primary), Matteo (Support)
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
NEIGHBORHOOD_DATA = "bau515od5155.csv"  # Property Prices by Neighborhood
BUILDING_AGE_DATA = "bau515od5156.csv"  # Property Prices by Building Age

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {PROCESSED_DATA_DIR}")

def load_datasets():
    """Load raw datasets."""
    try:
        # Load neighborhood dataset
        neighborhood_path = os.path.join(RAW_DATA_DIR, NEIGHBORHOOD_DATA)
        neighborhood_df = pd.read_csv(neighborhood_path)
        logger.info(f"Loaded neighborhood data: {neighborhood_path}")
        
        # Load building age dataset
        building_age_path = os.path.join(RAW_DATA_DIR, BUILDING_AGE_DATA)
        building_age_df = pd.read_csv(building_age_path)
        logger.info(f"Loaded building age data: {building_age_path}")
        
        return neighborhood_df, building_age_df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def clean_neighborhood_data(df):
    """
    Clean neighborhood dataset.
    
    TODO:
    - Handle missing values
    - Remove outliers (e.g., using IQR or z-score method)
    - Convert data types if necessary
    - Extract district information
    """
    logger.info("Cleaning neighborhood data...")
    
    # Add your cleaning code here
    # Example:
    # df = df.dropna(subset=['price', 'rooms'])
    # df = df[df['price'] > 0]
    
    return df

def clean_building_age_data(df):
    """
    Clean building age dataset.
    
    TODO:
    - Handle missing values
    - Remove outliers
    - Convert data types if necessary
    - Create age categories if needed
    """
    logger.info("Cleaning building age data...")
    
    # Add your cleaning code here
    
    return df

def merge_datasets(neighborhood_df, building_age_df):
    """
    Merge neighborhood and building age datasets if necessary.
    
    TODO:
    - Determine appropriate join keys
    - Handle conflicts or duplicates
    """
    logger.info("Merging datasets...")
    
    # Add your merging code here
    # Example:
    # merged_df = pd.merge(
    #     neighborhood_df, 
    #     building_age_df, 
    #     on=['year', 'rooms'], 
    #     how='inner'
    # )
    
    # For now, we'll just return them separately
    return neighborhood_df, building_age_df

def create_derived_features(df):
    """
    Create derived features.
    
    TODO:
    - Calculate price per room
    - Calculate year-over-year price changes
    - Create location clusters if needed
    """
    logger.info("Creating derived features...")
    
    # Add your feature engineering code here
    
    return df

def save_processed_data(neighborhood_df, building_age_df, merged_df=None):
    """Save processed datasets."""
    neighborhood_output = os.path.join(PROCESSED_DATA_DIR, "processed_neighborhood.csv")
    building_age_output = os.path.join(PROCESSED_DATA_DIR, "processed_building_age.csv")
    
    neighborhood_df.to_csv(neighborhood_output, index=False)
    logger.info(f"Saved processed neighborhood data: {neighborhood_output}")
    
    building_age_df.to_csv(building_age_output, index=False)
    logger.info(f"Saved processed building age data: {building_age_output}")
    
    if merged_df is not None:
        merged_output = os.path.join(PROCESSED_DATA_DIR, "processed_merged.csv")
        merged_df.to_csv(merged_output, index=False)
        logger.info(f"Saved processed merged data: {merged_output}")

def main():
    """Main data preparation pipeline."""
    start_time = datetime.now()
    logger.info("Starting data preparation pipeline")
    
    # Create directories
    create_directories()
    
    # Load raw datasets
    neighborhood_df, building_age_df = load_datasets()
    
    # Clean datasets
    neighborhood_df = clean_neighborhood_data(neighborhood_df)
    building_age_df = clean_building_age_data(building_age_df)
    
    # Merge datasets if necessary
    neighborhood_df, building_age_df = merge_datasets(neighborhood_df, building_age_df)
    
    # Create derived features
    neighborhood_df = create_derived_features(neighborhood_df)
    building_age_df = create_derived_features(building_age_df)
    
    # Save processed data
    save_processed_data(neighborhood_df, building_age_df)
    
    end_time = datetime.now()
    logger.info(f"Data preparation pipeline completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
