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

def
