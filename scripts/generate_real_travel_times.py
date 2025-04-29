"""
Script to generate real travel times for Zurich neighborhoods using Google Maps API.
This script fetches actual travel times between neighborhoods and key destinations.
"""

import os
import json
import time
import requests
import pandas as pd
import argparse

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
        "Albisrieden": "47.3783,8.4893",
        "Sihlfeld": "47.3720,8.5147",
        "Friesenberg": "47.3663,8.5030",
        "Leimbach": "47.3407,8.5124",
        "Wollishofen": "47.3510,8.5301",
        "Enge": "47.3650,8.5288",
        "Wiedikon": "47.3666,8.5182",
        "Hard": "47.3845,8.5090",
        "Unterstrass": "47.3887,8.5400",
        "Oberstrass": "47.3860,8.5487"
    }

def get_destination_coordinates():
    """
    Get coordinates for key destinations in Zurich.
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

def generate_travel_times(api_key, travel_mode="transit"):
    """
    Generate travel times for neighborhoods to key destinations using Google Maps API.
    
    Args:
        api_key (str): Google Maps API key
        travel_mode (str): Travel mode to use (transit, walking, driving, bicycling)
    
    Returns:
        dict: Travel times for each neighborhood to each destination
    """
    print(f"Generating travel time data using Google Maps API with mode: {travel_mode}...")
    
    # Get neighborhoods and coordinates
    neighborhoods = get_zurich_neighborhoods()
    neighborhood_coords = get_neighborhood_coordinates()
    destination_coords = get_destination_coordinates()
    
    # Create travel times dictionary with nested structure by travel mode
    travel_times = {}
    
    # Counter for API calls to avoid rate limits
    api_calls = 0
    
    for neighborhood in neighborhoods:
        # Skip neighborhoods we don't have coordinates for
        if neighborhood not in neighborhood_coords:
            print(f"Skipping {neighborhood} - no coordinates available")
            continue
        
        # Initialize neighborhood dictionary if not exists
        if neighborhood not in travel_times:
            travel_times[neighborhood] = {}
        
        # Initialize travel mode dictionary if not exists
        if travel_mode not in travel_times[neighborhood]:
            travel_times[neighborhood][travel_mode] = {}
        
        origin_coords = neighborhood_coords[neighborhood]
        
        for destination, dest_coords in destination_coords.items():
            # Add delay to avoid hitting API rate limits
            if api_calls > 0 and api_calls % 10 == 0:
                print(f"Pausing for 2 seconds after {api_calls} API calls...")
                time.sleep(2)
            
            # Get travel time using API
            travel_time = get_travel_time(origin_coords, dest_coords, api_key, mode=travel_mode)
            
            if travel_time is not None:
                travel_times[neighborhood][travel_mode][destination] = travel_time
                print(f"Travel time from {neighborhood} to {destination} by {travel_mode}: {travel_time} minutes")
            else:
                # Fallback to estimated travel time
                print(f"Using fallback travel time for {neighborhood} to {destination}")
                # Different fallbacks based on travel mode
                if travel_mode == "transit":
                    fallback_time = 30
                elif travel_mode == "walking":
                    fallback_time = 60
                elif travel_mode == "bicycling":
                    fallback_time = 25
                else:  # driving
                    fallback_time = 20
                
                travel_times[neighborhood][travel_mode][destination] = fallback_time
            
            api_calls += 1
    
    # Check if existing travel times file exists
    if os.path.exists("data/processed/travel_times.json"):
        try:
            # Load existing travel times
            with open("data/processed/travel_times.json", "r") as f:
                existing_travel_times = json.load(f)
            
            # Merge new travel times with existing ones
            for neighborhood in travel_times:
                if neighborhood not in existing_travel_times:
                    existing_travel_times[neighborhood] = {}
                
                for mode in travel_times[neighborhood]:
                    existing_travel_times[neighborhood][mode] = travel_times[neighborhood][mode]
            
            travel_times = existing_travel_times
        except Exception as e:
            print(f"Error merging with existing travel times: {e}")
    
    # Save travel times to file
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/travel_times.json", "w") as f:
        json.dump(travel_times, f, indent=2)
    
    print(f"Travel times saved to data/processed/travel_times.json")
    print(f"Total API calls made: {api_calls}")
    
    return travel_times

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate travel times for Zurich neighborhoods")
    parser.add_argument("--api-key", required=True, help="Google Maps API key")
    parser.add_argument("--mode", default="transit", choices=["transit", "walking", "driving", "bicycling"], 
                       help="Travel mode (default: transit)")
    
    args = parser.parse_args()
    
    # Generate travel times
    generate_travel_times(args.api_key, args.mode)

if __name__ == "__main__":
    main()
