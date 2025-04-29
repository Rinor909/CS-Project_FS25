import os
import json
import time
import requests
import pandas as pd

# You would need to set this to your actual API key
GOOGLE_MAPS_API_KEY = "AIzaSyBpOKo1b1LN-kMf5mQfEZyeqzIl2ddNTjk"

def get_zurich_neighborhoods():
    """
    Get a list of Zurich neighborhoods from the dataset.
    """
    try:
        if os.path.exists("data/raw/bau515od5155.csv"):
            df = pd.read_csv("data/raw/bau515od5155.csv")
            # Extract unique neighborhoods from the RaumLang column
            # Assuming RaumLang contains the neighborhood names
            neighborhoods = df["RaumLang"].unique().tolist()
            return neighborhoods
        else:
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

def get_travel_time(origin, destination, mode="transit"):
    """
    Get travel time between two locations using Google Maps Distance Matrix API.
    
    Args:
        origin (str): Origin coordinates in format "latitude,longitude"
        destination (str): Destination coordinates in format "latitude,longitude"
        mode (str): Travel mode (driving, walking, bicycling, transit)
        
    Returns:
        int: Travel time in minutes
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    params = {
        "origins": origin,
        "destinations": destination,
        "mode": mode,
        "key": GOOGLE_MAPS_API_KEY
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

def generate_travel_times():
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
            travel_time = get_travel_time(origin_coords, dest_coords)
            
            if travel_time is not None:
                travel_times[neighborhood][destination] = travel_time
                print(f"Travel time from {neighborhood} to {destination}: {travel_time} minutes")
            else:
                # Fallback to estimated travel time
                print(f"Using fallback travel time for {neighborhood} to {destination}")
                travel_times[neighborhood][destination] = 30  # Default 30 minutes
            
            api_calls += 1
    
    # Save travel times to file
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/travel_times.json", "w") as f:
        json.dump(travel_times, f, indent=2)
    
    print(f"Travel times saved to data/processed/travel_times.json")
    print(f"Total API calls made: {api_calls}")
    
    return travel_times

if __name__ == "__main__":
    # Check if API key is set
    if GOOGLE_MAPS_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: You need to set your Google Maps API key in the script")
        print("Get an API key from: https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
        exit(1)
    
    # Generate travel times
    generate_travel_times()
