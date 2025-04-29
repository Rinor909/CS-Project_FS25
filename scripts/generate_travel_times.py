import pandas as pd
import numpy as np
import os
import sys
import json
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Zurich key destinations
KEY_DESTINATIONS = {
    "Hauptbahnhof": {"lat": 47.3783, "lon": 8.5402},
    "ETH Zurich": {"lat": 47.3763, "lon": 8.5475},
    "Zurich Airport": {"lat": 47.4502, "lon": 8.5618},
    "Bahnhofstrasse": {"lat": 47.3708, "lon": 8.5392},
}

# Approximate neighborhood coordinates
# This is a placeholder - in a real app, you would have actual geocoordinates for each neighborhood
NEIGHBORHOOD_COORDS = {
    "Zurich (Kreis 1)": {"lat": 47.3769, "lon": 8.5417},
    "Zurich (Kreis 2)": {"lat": 47.3600, "lon": 8.5200},
    "Zurich (Kreis 3)": {"lat": 47.3786, "lon": 8.5124},
    "Zurich (Kreis 4)": {"lat": 47.3744, "lon": 8.5289},
    "Zurich (Kreis 5)": {"lat": 47.3853, "lon": 8.5253},
    "Zurich (Kreis 6)": {"lat": 47.3894, "lon": 8.5500},
    "Zurich (Kreis 7)": {"lat": 47.3672, "lon": 8.5681},
    "Zurich (Kreis 8)": {"lat": 47.3500, "lon": 8.5700},
    "Zurich (Kreis 9)": {"lat": 47.3900, "lon": 8.4900},
    "Zurich (Kreis 10)": {"lat": 47.4100, "lon": 8.5100},
    "Zurich (Kreis 11)": {"lat": 47.4200, "lon": 8.5300},
    "Zurich (Kreis 12)": {"lat": 47.3950, "lon": 8.5800},
}

def create_processed_data_dir():
    """Create processed data directory if it doesn't exist"""
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir

def get_all_neighborhoods():
    """Get all neighborhoods from the processed data"""
    processed_dir = os.path.join('data', 'processed')
    neighborhood_path = os.path.join(processed_dir, 'processed_neighborhood_data.csv')
    
    if not os.path.exists(neighborhood_path):
        raise FileNotFoundError(f"File not found: {neighborhood_path}. Run data_preparation.py first.")
    
    neighborhood_df = pd.read_csv(neighborhood_path)
    return sorted(neighborhood_df['neighborhood'].unique())

def simulate_travel_time(origin, destination):
    """
    Simulate a travel time calculation between two coordinates
    In a real app, this would call the Google Maps API
    
    Args:
        origin: Origin coordinates dict with lat/lon
        destination: Destination coordinates dict with lat/lon
        
    Returns:
        int: Estimated travel time in minutes
    """
    # Haversine distance calculation (simplified version)
    R = 6371  # Earth radius in kilometers
    
    lat1 = np.radians(origin["lat"])
    lon1 = np.radians(origin["lon"])
    lat2 = np.radians(destination["lat"])
    lon2 = np.radians(destination["lon"])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c  # Distance in km
    
    # Simulate travel time: approximately 3 minutes per km for public transport
    # Add some randomness to make it more realistic
    base_time = distance * 3
    variability = base_time * 0.2  # 20% variability
    
    travel_time = int(base_time + random.uniform(-variability, variability))
    
    # Ensure travel time is at least 5 minutes
    return max(5, travel_time)

def generate_travel_times():
    """
    Generate travel times from each neighborhood to key destinations
    
    Returns:
        dict: Nested dictionary of travel times
    """
    travel_times = {}
    neighborhoods = get_all_neighborhoods()
    
    print(f"Generating travel times for {len(neighborhoods)} neighborhoods to {len(KEY_DESTINATIONS)} destinations...")
    
    # Check if we have coordinates for all neighborhoods
    missing_coords = []
    for neighborhood in neighborhoods:
        if neighborhood not in NEIGHBORHOOD_COORDS:
            missing_coords.append(neighborhood)
            
            # Use a default coordinate (Zurich center) for missing neighborhoods
            NEIGHBORHOOD_COORDS[neighborhood] = {"lat": 47.3769, "lon": 8.5417}
    
    if missing_coords:
        print(f"Warning: Using default coordinates for {len(missing_coords)} neighborhoods:")
        print(", ".join(missing_coords[:5]) + ("..." if len(missing_coords) > 5 else ""))
    
    # Generate travel times for each neighborhood
    for neighborhood in neighborhoods:
        neighborhood_times = {}
        origin = NEIGHBORHOOD_COORDS[neighborhood]
        
        for dest_name, dest_coords in KEY_DESTINATIONS.items():
            travel_time = simulate_travel_time(origin, dest_coords)
            neighborhood_times[dest_name] = travel_time
        
        travel_times[neighborhood] = neighborhood_times
    
    print(f"Generated travel times for all neighborhoods")
    return travel_times

def save_travel_times(travel_times, processed_dir):
    """Save travel times to a JSON file"""
    travel_times_path = os.path.join(processed_dir, 'travel_times.json')
    
    with open(travel_times_path, 'w') as f:
        json.dump(travel_times, f, indent=2)
    
    print(f"Saved travel times to {travel_times_path}")
    
    # Also save as CSV for easier analysis
    csv_data = []
    for neighborhood, dest_times in travel_times.items():
        for destination, time in dest_times.items():
            csv_data.append({
                'neighborhood': neighborhood,
                'destination': destination,
                'travel_time_minutes': time
            })
    
    travel_times_csv_path = os.path.join(processed_dir, 'travel_times.csv')
    pd.DataFrame(csv_data).to_csv(travel_times_csv_path, index=False)
    print(f"Saved travel times CSV to {travel_times_csv_path}")

def main():
    """Main function to generate travel times"""
    print("Starting travel time generation...")
    
    # Create processed data directory
    processed_dir = create_processed_data_dir()
    
    try:
        # Generate travel times
        travel_times = generate_travel_times()
        
        # Save travel times
        save_travel_times(travel_times, processed_dir)
        
        print("Travel time generation completed successfully!")
        
    except Exception as e:
        print(f"Error during travel time generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()