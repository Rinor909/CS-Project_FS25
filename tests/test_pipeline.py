"""
Test Pipeline for Zurich Real Estate Price Prediction
----------------------------------------------------
Purpose: Test the data processing and model prediction pipeline

Tasks:
1. Test data loading
2. Test data preprocessing
3. Test model loading
4. Test price prediction

Owner: Rinor (Primary)
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import pickle

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from scripts.data_preparation import load_datasets, clean_neighborhood_data, clean_building_age_data
from app.utils import load_model, prepare_prediction_input, predict_price

class TestPipeline(unittest.TestCase):
    """Test cases for the data processing and model prediction pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.raw_data_dir = "../data/raw"
        self.processed_data_dir = "../data/processed"
        self.models_dir = "../models"
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def test_data_loading(self):
        """Test that data can be loaded correctly."""
        try:
            # This assumes that load_datasets has been implemented with the specified arguments
            neighborhood_df, building_age_df = load_datasets()
            
            # Check that dataframes are not empty
            self.assertFalse(neighborhood_df.empty, "Neighborhood dataframe is empty")
            self.assertFalse(building_age_df.empty, "Building age dataframe is empty")
            
            print("Data loading test passed")
        except Exception as e:
            self.fail(f"Data loading test failed: {e}")
    
    def test_data_cleaning(self):
        """Test that data cleaning functions work correctly."""
        try:
            # This assumes that load_datasets has been implemented with the specified arguments
            neighborhood_df, building_age_df = load_datasets()
            
            # Clean data
            cleaned_neighborhood = clean_neighborhood_data(neighborhood_df)
            cleaned_building_age = clean_building_age_data(building_age_df)
            
            # Check that dataframes are not empty after cleaning
            self.assertFalse(cleaned_neighborhood.empty, "Cleaned neighborhood dataframe is empty")
            self.assertFalse(cleaned_building_age.empty, "Cleaned building age dataframe is empty")
            
            # Check that there are no missing values in key columns
            # This will depend on your actual column names
            key_columns = ['HAMedianPreis']  # Update with your actual price column
            for col in key_columns:
                if col in cleaned_neighborhood.columns:
                    self.assertEqual(
                        cleaned_neighborhood[col].isna().sum(), 0,
                        f"Missing values in {col} column after cleaning"
                    )
            
            print("Data cleaning test passed")
        except Exception as e:
            self.fail(f"Data cleaning test failed: {e}")
    
    def test_model_loading(self):
        """Test that the model can be loaded correctly."""
        try:
            # Create a dummy model if it doesn't exist
            model_path = os.path.join(self.models_dir, "price_model.pkl")
            if not os.path.exists(model_path):
                print("No model found, creating dummy model for testing")
                dummy_model = {
                    'model': None,
                    'features': ['neighborhood', 'room_count', 'building_age', 'travel_time_hauptbahnhof']
                }
                with open(model_path, 'wb') as f:
                    pickle.dump(dummy_model, f)
            
            # Load model
            model, features = load_model()
            
            # Check that features is not None
            self.assertIsNotNone(features, "Model features are None")
            
            print("Model loading test passed")
        except Exception as e:
            self.fail(f"Model loading test failed: {e}")
    
    def test_prediction_input_preparation(self):
        """Test that prediction input can be prepared correctly."""
        try:
            # Prepare dummy input
            neighborhood = "Kreis 1"
            room_count = 3
            building_age = 40
            travel_times = {
                'Hauptbahnhof': 15,
                'ETH_Zurich': 20,
                'Zurich_Airport': 35,
                'Bahnhofstrasse': 10
            }
            
            # Prepare input
            input_data = prepare_prediction_input(neighborhood, room_count, building_age, travel_times)
            
            # Check that input_data is a DataFrame
            self.assertIsInstance(input_data, pd.DataFrame, "Input data is not a DataFrame")
            
            # Check that the DataFrame has at least one row
            self.assertGreater(len(input_data), 0, "Input data has no rows")
            
            print("Prediction input preparation test passed")
        except Exception as e:
            self.fail(f"Prediction input preparation test failed: {e}")
    
    def test_price_prediction(self):
        """Test that price prediction works correctly."""
        try:
            # Create a dummy model if it doesn't exist
            model_path = os.path.join(self.models_dir, "price_model.pkl")
            if not os.path.exists(model_path):
                print("No model found, creating dummy model for testing")
                
                # Create a dummy model that always returns 1,000,000
                class DummyModel:
                    def predict(self, X):
                        return np.array([1000000])
                
                dummy_model = {
                    'model': DummyModel(),
                    'features': ['neighborhood_factor', 'room_count', 'building_age', 'travel_time_hauptbahnhof']
                }
                with open(model_path, 'wb') as f:
                    pickle.dump(dummy_model, f)
            
            # Load model
            model, features = load_model()
            
            # Prepare dummy input (matching the feature names in the dummy model)
            input_data = pd.DataFrame({
                'neighborhood_factor': [0.5],
                'room_count': [3],
                'building_age': [40],
                'travel_time_hauptbahnhof': [15]
            })
            
            # Make prediction
            predicted_price = predict_price(model, features, input_data)
            
            # Check that prediction is a number
            self.assertIsNotNone(predicted_price, "Predicted price is None")
            
            # Check that prediction is positive
            self.assertGreater(predicted_price, 0, "Predicted price is not positive")
            
            print("Price prediction test passed")
        except Exception as e:
            self.fail(f"Price prediction test failed: {e}")

if __name__ == "__main__":
    unittest.main()
