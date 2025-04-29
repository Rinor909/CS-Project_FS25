"""
Setup script for Zurich Real Estate Price Prediction app.
This script creates directories, moves files, and generates travel time data.
"""

import os
import shutil
import json
import pickle
import numpy as np
import time
import requests
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def create_directories():
    """Create the necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_csv_files():
    """Move CSV files to data/raw if they exist in the current directory"""
    csv_files = ["bau515od5155.csv", "bau515od5156.csv"]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            destination = os.path.join("data/raw", csv_file)
            shutil.copy(csv_file, destination)
            print(f"Moved {csv_file} to {destination}")
        else:
            print(f"Warning: {csv_file} not found in current directory")

def get_api_key():
    """Get Google Maps API key from user"""
    print("\n===== Google Maps API Setup =====")
    print("To generate real travel time data, you need a Google Maps API key.")
    print("If you don't have one, you can get it from: https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
    print("Note: Using the Google Maps API may incur charges to your Google Cloud account.")
    
    use_api = input("Do you want to use Google Maps API for real travel time data? (y/n): ").strip().lower()
    
    if use_api == 'y':
        api_key = input("Enter your Google Maps API key: ").strip()
        return api_key
    else:
        return None

def get_zurich_neighborhoods():
    """
    Get a list of Zurich neighborhoods from the dataset.
    """
    try:
        if os.path.exists("data/raw/bau515od5155.csv"):
            df = pd.read_csv("data/raw/bau515od5155.csv")
            # Extract unique neighborhoods from the RaumLang column
            if 'RaumLang' in df.columns:
                neighborhoods = df["RaumLang"].unique().tolist()
                return neighborhoods
        
        # Fallback neighborhoods
        return [
            "Altstadt", "Escher Wyss", "Gewerbeschule", "Hochschulen", 
            "Höngg", "Oerlikon", "Seebach", "Altstetten", "Albisrieden"
        ]
    except Exception as e:
        print(f"Error reading neighborhood data: {e}")
        return [
            "Altstadt", "Escher Wyss", "Gewerbeschule", "Hochschulen", 
            "Höngg", "Oerlikon"
        ]

def get_neighborhood_coordinates():
    """
    Get approximate coordinates for each Zurich neighborhood.
    In a real implementation, you would want to get more accurate coordinates.
    """
    return {
        "Altstadt": "47.3723,8.5423",
        "Escher Wyss": "47.3914,8.5152",
        "Gewerbeschule": "47.3845,8.5293",
        "Hochschulen": "47.3764,8.5468",
        "Höngg": "47.4036,8.4894",
        "Oerlikon": "47.4119,8.5450",
        "Seebach": "47.4231,8.5392",
        "Altstetten": "47.3889,8.4833",
        "Albisrieden": "47.3783,8.4893"
    }

def get_destination_coordinates():
    """
    Get approximate coordinates for key destinations in Zurich.
    """
    return {
        "Hauptbahnhof": "47.3783,8.5403",
        "ETH Zurich": "47.3761,8.5458",
        "Zurich Airport": "47.4502,8.5582",
        "Bahnhofstrasse": "47.3725,8.5380"
    }

def get_travel_time(origin, destination, api_key, mode="transit"):
    """
    Get travel time between two locations using Google Maps Distance Matrix API.
    
    Args:
        origin (str): Origin coordinates in format "latitude,longitude"
        destination (str): Destination coordinates in format "latitude,longitude"
        api_key (str): Google Maps API key
        mode (str): Travel mode (driving, walking, bicycling, transit)
        
    Returns:
        int: Travel time in minutes
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    params = {
        "origins": origin,
        "destinations": destination,
        "mode": mode,
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["status"] == "OK" and data["rows"][0]["elements"][0]["status"] == "OK":
            # Get travel time in seconds and convert to minutes
            travel_time_seconds = data["rows"][0]["elements"][0]["duration"]["value"]
            travel_time_minutes = round(travel_time_seconds / 60)
            return travel_time_minutes
        else:
            print(f"Error getting travel time: {data}")
            return None
    except Exception as e:
        print(f"Error calling Google Maps API: {e}")
        return None

def generate_travel_times_with_api(api_key):
    """
    Generate travel times for neighborhoods to key destinations using Google Maps API.
    """
    print("Generating travel time data using Google Maps API...")
    
    # Get neighborhoods and coordinates
    neighborhoods = get_zurich_neighborhoods()
    neighborhood_coords = get_neighborhood_coordinates()
    destination_coords = get_destination_coordinates()
    
    # Create travel times dictionary
    travel_times = {}
    
    # Counter for API calls to avoid rate limits
    api_calls = 0
    
    for neighborhood in neighborhoods:
        # Skip neighborhoods we don't have coordinates for
        if neighborhood not in neighborhood_coords:
            print(f"Skipping {neighborhood} - no coordinates available")
            continue
        
        travel_times[neighborhood] = {}
        origin_coords = neighborhood_coords[neighborhood]
        
        for destination, dest_coords in destination_coords.items():
            # Add delay to avoid hitting API rate limits
            if api_calls > 0 and api_calls % 10 == 0:
                print(f"Pausing for 2 seconds after {api_calls} API calls...")
                time.sleep(2)
            
            # Get travel time using API
            travel_time = get_travel_time(origin_coords, dest_coords, api_key)
            
            if travel_time is not None:
                travel_times[neighborhood][destination] = travel_time
                print(f"Travel time from {neighborhood} to {destination}: {travel_time} minutes")
            else:
                # Fallback to estimated travel time
                print(f"Using fallback travel time for {neighborhood} to {destination}")
                travel_times[neighborhood][destination] = 30  # Default 30 minutes
            
            api_calls += 1
    
    # Save travel times to file
    with open("data/processed/travel_times.json", "w") as f:
        json.dump(travel_times, f, indent=2)
    
    print(f"Travel times saved to data/processed/travel_times.json")
    print(f"Total API calls made: {api_calls}")
    
    return travel_times

def generate_synthetic_travel_times():
    """Generate synthetic travel time data"""
    print("Generating synthetic travel time data...")
    
    # Sample neighborhoods
    neighborhoods = get_zurich_neighborhoods()
    
    # Sample destinations
    destinations = ["Hauptbahnhof", "ETH Zurich", "Zurich Airport", "Bahnhofstrasse"]
    
    # Create sample travel time data
    travel_times = {}
    for neighborhood in neighborhoods:
        travel_times[neighborhood] = {}
        
        # Generate times based on rough geographic knowledge of Zurich
        if neighborhood in ["Altstadt", "Hochschulen"]:
            # Central neighborhoods
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(5, 15)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(5, 15)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(25, 40)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(5, 15)
        elif neighborhood in ["Oerlikon", "Seebach"]:
            # Northern neighborhoods (closer to airport)
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(15, 25)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(15, 25)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(10, 20)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(20, 30)
        else:
            # Other neighborhoods
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(15, 30)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(20, 35)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(30, 50)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(20, 40)
    
    # Save travel time data
    with open("data/processed/travel_times.json", "w") as f:
        json.dump(travel_times, f, indent=2)
    
    print(f"Synthetic travel times saved to data/processed/travel_times.json")

def create_model():
    """Create a simple price prediction model"""
    print("Creating price prediction model...")
    
    # Create a simple random forest model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create synthetic training data
    X = np.random.rand(100, 4)  # [neighborhood_code, rooms, building_age, travel_time]
    y = 1000000 + 500000 * X[:, 0] + 200000 * X[:, 1] - 10000 * X[:, 2] - 5000 * X[:, 3]
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    with open("models/price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to models/price_model.pkl")

def main():
    print("\n===== Zurich Real Estate Price Prediction App Setup =====\n")
    
    # Create directories
    create_directories()
    
    # Move CSV files
    move_csv_files()
    
    # Generate travel time data if it doesn't exist
    if not os.path.exists("data/processed/travel_times.json"):
        api_key = get_api_key()
        
        if api_key:
            try:
                # Try to import requests
                import requests
                generate_travel_times_with_api(api_key)
            except ImportError:
                print("Error: 'requests' module not found. Installing...")
                os.system("pip install requests")
                try:
                    import requests
                    generate_travel_times_with_api(api_key)
                except Exception as e:
                    print(f"Error with API calls: {e}")
                    print("Falling back to synthetic travel time data...")
                    generate_synthetic_travel_times()
        else:
            print("No API key provided. Generating synthetic travel time data...")
            generate_synthetic_travel_times()
    else:
        print("Travel time data already exists at data/processed/travel_times.json")
    
    # Create model if it doesn't exist
    if not os.path.exists("models/price_model.pkl"):
        create_model()
    else:
        print("Price prediction model already exists at models/price_model.pkl")
    
    print("\n===== Setup complete! =====")
    print("\nTo run the Streamlit app, use the command:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
