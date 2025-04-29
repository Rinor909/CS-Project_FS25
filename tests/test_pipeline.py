import os
import sys
import unittest
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.utils import load_data, preprocess_data
from scripts.data_preparation import load_neighborhood_data, load_building_age_data


class TestDataPipeline(unittest.TestCase):
    """Test cases for the data pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.neighborhood_path = os.path.join('data', 'raw', 'bau515od5155.csv')
        self.building_age_path = os.path.join('data', 'raw', 'bau515od5156.csv')
        
        # Check if data files exist
        self.data_exists = os.path.exists(self.neighborhood_path) and os.path.exists(self.building_age_path)
        
        if not self.data_exists:
            print(f"Warning: Test data files not found at {self.neighborhood_path} or {self.building_age_path}")
    
    def test_data_files_exist(self):
        """Test that data files exist"""
        self.assertTrue(os.path.exists(self.neighborhood_path), f"Neighborhood data file not found at {self.neighborhood_path}")
        self.assertTrue(os.path.exists(self.building_age_path), f"Building age data file not found at {self.building_age_path}")
    
    def test_load_data(self):
        """Test that data loading function works"""
        if not self.data_exists:
            self.skipTest("Data files not found")
        
        # Load neighborhood data
        neighborhood_df = load_data(self.neighborhood_path)
        self.assertIsInstance(neighborhood_df, pd.DataFrame, "Neighborhood data should be a DataFrame")
        self.assertGreater(len(neighborhood_df), 0, "Neighborhood data should not be empty")
        
        # Load building age data
        building_age_df = load_data(self.building_age_path)
        self.assertIsInstance(building_age_df, pd.DataFrame, "Building age data should be a DataFrame")
        self.assertGreater(len(building_age_df), 0, "Building age data should not be empty")
    
    def test_preprocess_neighborhood_data(self):
        """Test neighborhood data preprocessing"""
        if not self.data_exists:
            self.skipTest("Data files not found")
        
        # Load and preprocess neighborhood data
        try:
            neighborhood_df = load_neighborhood_data()
            self.assertIsInstance(neighborhood_df, pd.DataFrame, "Processed neighborhood data should be a DataFrame")
            self.assertGreater(len(neighborhood_df), 0, "Processed neighborhood data should not be empty")
            
            # Check required columns
            required_cols = ['year', 'neighborhood', 'room_count', 'median_price']
            for col in required_cols:
                self.assertIn(col, neighborhood_df.columns, f"Column {col} should be in processed neighborhood data")
        except Exception as e:
            self.fail(f"Preprocessing neighborhood data failed with error: {str(e)}")
    
    def test_preprocess_building_age_data(self):
        """Test building age data preprocessing"""
        if not self.data_exists:
            self.skipTest("Data files not found")
        
        # Load and preprocess building age data
        try:
            building_age_df = load_building_age_data()
            self.assertIsInstance(building_age_df, pd.DataFrame, "Processed building age data should be a DataFrame")
            self.assertGreater(len(building_age_df), 0, "Processed building age data should not be empty")
            
            # Check required columns
            required_cols = ['year', 'building_age', 'room_count', 'median_price']
            for col in required_cols:
                self.assertIn(col, building_age_df.columns, f"Column {col} should be in processed building age data")
        except Exception as e:
            self.fail(f"Preprocessing building age data failed with error: {str(e)}")
    
    def test_data_pipeline(self):
        """Test the complete data pipeline"""
        if not self.data_exists:
            self.skipTest("Data files not found")
        
        try:
            # Load raw data
            neighborhood_df = load_data(self.neighborhood_path)
            building_age_df = load_data(self.building_age_path)
            
            # Process data
            processed_data, neighborhoods, room_counts, building_ages, latest_year = preprocess_data(
                neighborhood_df, building_age_df
            )
            
            # Check outputs
            self.assertIsInstance(processed_data, pd.DataFrame, "Processed data should be a DataFrame")
            self.assertIsInstance(neighborhoods, list, "Neighborhoods should be a list")
            self.assertIsInstance(room_counts, list, "Room counts should be a list")
            self.assertIsInstance(building_ages, list, "Building ages should be a list")
            self.assertIsInstance(latest_year, (int, np.integer), "Latest year should be an integer")
            
            # Check for non-empty data
            self.assertGreater(len(processed_data), 0, "Processed data should not be empty")
            self.assertGreater(len(neighborhoods), 0, "Neighborhoods list should not be empty")
            self.assertGreater(len(room_counts), 0, "Room counts list should not be empty")
            self.assertGreater(len(building_ages), 0, "Building ages list should not be empty")
        except Exception as e:
            self.fail(f"Data pipeline failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()
