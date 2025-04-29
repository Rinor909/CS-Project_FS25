"""
Travel Time Generation Script for Zurich Real Estate Price Prediction
--------------------------------------------------------------------
Purpose: Generate travel time data from neighborhoods to key destinations

Tasks:
1. Connect to Google Maps API (or Comparis)
2. Calculate travel times from each neighborhood to key destinations
3. Cache results to avoid API limits
4. Export travel time data for model training

Key destinations:
- Hauptbahnhof (Main Train Station)
- ETH Zurich
- Zurich Airport
- Bahnhofstrasse

Owner: Rinor (Primary), Matteo (Support)
"""

import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
PROCESSED_DATA_DIR = "../data/processed"
PROCESSED_NEIGHBORHOOD_DATA = "processed_neighborhood.csv"
CACHE_DIR = "../data/cache"
CACHE_FILE = "travel_times_cache.json"
OUTPUT_FILE = "neighborhood_travel_times.csv"

# Define key destinations (coordinates)
KEY_DESTINATIONS = {
    "Hauptbahnhof": {"lat": 47.3782, "lng": 8.5401},
    "ETH_Zurich": {"lat": 47.3763, "lng": 8.5475},
    "Zurich_Airport": {"lat": 47.4502, "lng": 8.5614},
    "Bahnhofstrasse": {"lat": 47.3723, "lng": 8.5390}
}

# Google Maps API key (to be set as environment variable)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_API_KEY_HERE")

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Ensured directories exist: {PROCESSED_DATA_DIR}, {CACHE_DIR}")

def load_cache():
    """Load existing cache of travel times."""
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached travel times")
            return cache
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    else:
        logger.info("No cache file found, creating new cache")
        return {}

def save_cache(cache):
    """Save cache of travel times."""
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        logger.info(f"Saved {len(cache)} travel times to cache")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def get_neighborhood_coordinates():
    """
    Get coordinates for each neighborhood.
    
    TODO:
    - Either load from processed data or define manually
    - Create a mapping of neighborhood names to coordinates
    """
    # This is a placeholder - in a real implementation, you would:
    # 1. Either have these coordinates in your dataset
    # 2. Or use geocoding to get them
    # 3. Or manually define them
    
    # Example placeholder data - replace with real data:
    neighborhood_coords = {
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
    
    logger.info(f"Loaded coordinates for {len(neighborhood_coords)} neighborhoods")
    return neighborhood_coords

def get_travel_time_from_api(origin, destination, mode="transit"):
    """
    Get travel time from Google Maps API.
    
    Parameters:
    - origin: dict with lat, lng
    - destination: dict with lat, lng
    - mode: transit, driving, walking, or bicycling
    
    Returns:
    - Travel time in minutes
    """
    # Google Maps Directions API endpoint
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Prepare parameters
    params = {
        "origin": f"{origin['lat']},{origin['lng']}",
        "destination": f"{destination['lat']},{destination['lng']}",
        "mode": mode,
        "key": API_KEY
    }
    
    try:
        # This is where you would make the actual API call
        # For this template, we'll just return dummy data
        # response = requests.get(url, params=params)
        # response.raise_for_status()
        # data = response.json()
        # if data["status"] == "OK":
        #     # Extract duration in minutes
        #     duration_seconds = data["routes"][0]["legs"][0]["duration"]["value"]
        #     duration_minutes = duration_seconds / 60
        #     return round(duration_minutes)
        # else:
        #     logger.error(f"API error: {data['status']}")
        #     return None
        
        # Dummy implementation - replace with actual API call in production
        import random
        time.sleep(0.1)  # Simulate API call delay
        return random.randint(5, 60)  # Random duration between 5-60 minutes
    
    except Exception as e:
        logger.error(f"Error getting travel time: {e}")
        return None

def generate_travel_times():
    """Generate travel times from neighborhoods to key destinations."""
    # Load cache
    cache = load_cache()
    
    # Get neighborhood coordinates
    neighborhood_coords = get_neighborhood_coordinates()
    
    # Prepare dataframe to store results
    results = []
    
    # Calculate travel times for each neighborhood to each destination
    for neighborhood, origin in neighborhood_coords.items():
        logger.info(f"Processing travel times for {neighborhood}")
        
        for dest_name, destination in KEY_DESTINATIONS.items():
            # Create cache key
            cache_key = f"{neighborhood}_{dest_name}_transit"
            
            # Check if we have this in cache
            if cache_key in cache:
                travel_time = cache[cache_key]
                logger.debug(f"Using cached value for {cache_key}: {travel_time} minutes")
            else:
                # Call API
                travel_time = get_travel_time_from_api(origin, destination)
                
                # Store in cache
                if travel_time is not None:
                    cache[cache_key] = travel_time
                    logger.debug(f"Added to cache: {cache_key}: {travel_time} minutes")
                
                # Add a delay to avoid hitting API rate limits
                time.sleep(0.5)
            
            # Add to results
            if travel_time is not None:
                results.append({
                    "neighborhood": neighborhood,
                    "destination": dest_name,
                    "travel_time_minutes": travel_time
                })
    
    # Save updated cache
    save_cache(cache)
    
    # Convert results to DataFrame
    travel_times_df = pd.DataFrame(results)
    logger.info(f"Generated {len(travel_times_df)} travel time entries")
    
    return travel_times_df

def save_travel_times(travel_times_df):
    """Save travel times to CSV."""
    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)
    travel_times_df.to_csv(output_path, index=False)
    logger.info(f"Saved travel times to {output_path}")

def main():
    """Main travel time generation pipeline."""
    start_time = datetime.now()
    logger.info("Starting travel time generation")
    
    # Create directories
    create_directories()
    
    # Generate travel times
    travel_times_df = generate_travel_times()
    
    # Save results
    save_travel_times(travel_times_df)
    
    end_time = datetime.now()
    logger.info(f"Travel time generation completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
