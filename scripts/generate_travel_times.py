import pandas as pd
import numpy as np
import os
import json
import time
import requests
import sys

# Function to read API key directly from GitHub
def get_api_key_from_github():
    secrets_url = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/.streamlit/secrets.toml'
    try:
        response = requests.get(secrets_url)
        if response.status_code == 200:
            content = response.text
            for line in content.split('\n'):
                if line.startswith('GOOGLE_MAPS_API_KEY'):
                    api_key = line.split('=')[1].strip().strip('"').strip("'")
                    return api_key
        else:
            print(f"Failed to get secrets file: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error getting API key from GitHub: {e}")
    return None

# Try to get API key from GitHub
print("Loading API key from GitHub...")
GOOGLE_MAPS_API_KEY = get_api_key_from_github()

# Fallback to environment variable if GitHub fails
if not GOOGLE_MAPS_API_KEY:
    print("Trying environment variable...")
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# Check if we have a key
if not GOOGLE_MAPS_API_KEY:
    print("ERROR: No Google Maps API key found.")
    print("Please make sure your API key is correctly set in:")
    print("https://github.com/Rinor909/zurich-real-estate/blob/main/.streamlit/secrets.toml")
    sys.exit(1)
else:
    print(f"Successfully loaded API key: {GOOGLE_MAPS_API_KEY[:5]}...{GOOGLE_MAPS_API_KEY[-5:]}")

# Use direct GitHub URLs for data loading
print("Loading quartier data from GitHub...")
url_quartier = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/quartier_processed.csv'

try:
    df_quartier = pd.read_csv(url_quartier)
    quartiere = df_quartier['Quartier'].unique()
    print(f"Loaded {len(quartiere)} unique neighborhoods.")
except Exception as e:
    print(f"Error loading quartier data from GitHub: {e}")
    print("ERROR: Could not load quartier data.")
    sys.exit(1)

# Define output directory
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# Wichtige Zielorte in Zürich
zielorte = {
    'Hauptbahnhof': 'Zürich Hauptbahnhof, Zürich, Schweiz',
    'ETH': 'ETH Zürich, Rämistrasse 101, 8092 Zürich, Schweiz',
    'Flughafen': 'Flughafen Zürich, Kloten, Schweiz',
    'Bahnhofstrasse': 'Bahnhofstrasse, Zürich, Schweiz'
}

# Tatsächliche zentrale Koordinaten für jedes Quartier
quartier_koordinaten = {
    'Hottingen': {'lat': 47.3692, 'lng': 8.5631},
    'Fluntern': {'lat': 47.3809, 'lng': 8.5629},
    'Unterstrass': {'lat': 47.3864, 'lng': 8.5419},
    'Oberstrass': {'lat': 47.3889, 'lng': 8.5481},
    'Rathaus': {'lat': 47.3716, 'lng': 8.5428},
    'Lindenhof': {'lat': 47.3728, 'lng': 8.5408},
    'City': {'lat': 47.3752, 'lng': 8.5385},
    'Seefeld': {'lat': 47.3600, 'lng': 8.5532},
    'Mühlebach': {'lat': 47.3638, 'lng': 8.5471},
    'Witikon': {'lat': 47.3610, 'lng': 8.5881},
    'Hirslanden': {'lat': 47.3624, 'lng': 8.5705},
    'Enge': {'lat': 47.3628, 'lng': 8.5288},
    'Wollishofen': {'lat': 47.3489, 'lng': 8.5266},
    'Leimbach': {'lat': 47.3279, 'lng': 8.5098},
    'Friesenberg': {'lat': 47.3488, 'lng': 8.5035},
    'Alt-Wiedikon': {'lat': 47.3652, 'lng': 8.5158},
    'Sihlfeld': {'lat': 47.3742, 'lng': 8.5072},
    'Albisrieden': {'lat': 47.3776, 'lng': 8.4842},
    'Altstetten': {'lat': 47.3917, 'lng': 8.4876},
    'Höngg': {'lat': 47.4023, 'lng': 8.4976},
    'Wipkingen': {'lat': 47.3930, 'lng': 8.5253},
    'Affoltern': {'lat': 47.4230, 'lng': 8.5047},
    'Oerlikon': {'lat': 47.4126, 'lng': 8.5487},
    'Seebach': {'lat': 47.4258, 'lng': 8.5422},
    'Saatlen': {'lat': 47.4087, 'lng': 8.5742},
    'Schwamendingen-Mitte': {'lat': 47.4064, 'lng': 8.5648},
    'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841},
    'Ganze Stadt': {'lat': 47.3769, 'lng': 8.5417},
    'Kreis 1': {'lat': 47.3732, 'lng': 8.5413},
    'Kreis 2': {'lat': 47.3559, 'lng': 8.5277},
    'Kreis 3': {'lat': 47.3682, 'lng': 8.5097},
    'Kreis 4': {'lat': 47.3767, 'lng': 8.5257},
    'Kreis 5': {'lat': 47.3875, 'lng': 8.5295},
    'Kreis 6': {'lat': 47.3846, 'lng': 8.5498},
    'Kreis 7': {'lat': 47.3637, 'lng': 8.5751},
    'Kreis 8': {'lat': 47.3560, 'lng': 8.5513},
    'Kreis 9': {'lat': 47.3796, 'lng': 8.4882},
    'Kreis 10': {'lat': 47.4088, 'lng': 8.5253},
    'Kreis 11': {'lat': 47.4173, 'lng': 8.5456},
    'Kreis 12': {'lat': 47.3985, 'lng': 8.5761},
    'Hochschulen': {'lat': 47.3743, 'lng': 8.5482},
    'Langstrasse': {'lat': 47.3800, 'lng': 8.5300},
    'Wurde-Furrer': {'lat': 47.3791, 'lng': 8.5261},
    'Escher-Wyss': {'lat': 47.3888, 'lng': 8.5219},
    'Gewerbeschule': {'lat': 47.3850, 'lng': 8.5311},
    'Hard': {'lat': 47.3832, 'lng': 8.5198}
}

# Für fehlende Quartiere Standardwerte hinzufügen
missing_quartiers = []
for quartier in quartiere:
    if quartier not in quartier_koordinaten:
        missing_quartiers.append(quartier)
        # Setze Default-Koordinaten für Zürich Zentrum
        quartier_koordinaten[quartier] = {'lat': 47.3769, 'lng': 8.5417}

if missing_quartiers:
    print(f"Warning: Used default coordinates for {len(missing_quartiers)} neighborhoods: {', '.join(missing_quartiers[:5])}{' and more...' if len(missing_quartiers) > 5 else ''}")

# Funktion zur Berechnung der Reisezeit mit Google Maps API
def get_travel_time(origin, destination, mode='transit'):
    """
    Berechnet die Reisezeit zwischen zwei Orten mit Google Maps API
    
    Args:
        origin (dict): {'lat': float, 'lng': float} - Ursprungskoordinaten
        destination (str): Zieladresse als String
        mode (str): Transportmittel ('transit', 'driving', 'walking', 'bicycling')
        
    Returns:
        float: Reisezeit in Minuten
    """
    # Cache file path
    cache_file = os.path.join(processed_dir, 'travel_time_cache.json')
    
    # Cache laden, falls vorhanden
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted cache file
            print(f"Warning: Corrupted cache file. Creating new cache.")
            cache = {}
    else:
        cache = {}
    
    # Cache-Key erstellen
    cache_key = f"{origin['lat']},{origin['lng']}_TO_{destination}_{mode}"
    
    # Wenn Ergebnis im Cache, aus Cache zurückgeben
    if cache_key in cache:
        return cache[cache_key]
    
    # URL für Google Maps Directions API
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Parameter für die Anfrage
    params = {
        'origin': f"{origin['lat']},{origin['lng']}",
        'destination': destination,
        'mode': mode,
        'key': GOOGLE_MAPS_API_KEY,
        'departure_time': 'now',
        'alternatives': 'false'
    }
    
    # Anfrage senden
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Prüfen, ob die Anfrage erfolgreich war
        if data['status'] == 'OK':
            # Reisezeit aus der ersten Route extrahieren
            route = data['routes'][0]
            leg = route['legs'][0]
            duration_seconds = leg['duration']['value']
            duration_minutes = duration_seconds / 60
            
            # Im Cache speichern
            cache[cache_key] = duration_minutes
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
                
            return duration_minutes
        else:
            print(f"API Error: {data['status']}")
            if 'error_message' in data:
                print(f"Error details: {data['error_message']}")
            
            if data['status'] == 'REQUEST_DENIED':
                print("ERROR: API request denied. Check your API key.")
                sys.exit(1)
                
            print("ERROR: Failed to get travel time from API.")
            return None
    except Exception as e:
        print(f"ERROR: Could not get travel time data: {e}")
        return None

if __name__ == "__main__":
    # DataFrame für Reisezeiten erstellen
    travel_times = []
    
    if len(quartiere) == 0:
        print("ERROR: No neighborhoods found.")
        sys.exit(1)

    # Limit the number of neighborhoods to process if too many
    max_quartiere = 100  # Set a reasonable limit
    if len(quartiere) > max_quartiere:
        print(f"Warning: Large number of neighborhoods ({len(quartiere)}). Processing the first {max_quartiere}.")
        quartiere = quartiere[:max_quartiere]

    # Progress counter
    total_calculations = len(quartiere) * len(zielorte) * 2  # 2 transport modes
    processed = 0
    
    print(f"Starting travel time calculations for {len(quartiere)} neighborhoods to {len(zielorte)} destinations...")
    print(f"Total calculations to perform: {total_calculations}")

    # Verify API key works by testing one calculation
    test_quartier = quartiere[0]
    test_origin = quartier_koordinaten.get(test_quartier)
    test_result = get_travel_time(test_origin, zielorte['Hauptbahnhof'], 'transit')
    
    if test_result is None:
        print("ERROR: Initial API test failed. Please check your API key and connection.")
        sys.exit(1)
    else:
        print(f"API test successful. Travel time from {test_quartier} to Hauptbahnhof: {test_result:.1f} minutes.")

    # Für jedes Quartier die Reisezeiten zu allen Zielorten berechnen
    for quartier in quartiere:
        origin = quartier_koordinaten.get(quartier)
        if not origin:
            print(f"No coordinates found for {quartier}. Skipping.")
            continue
            
        print(f"Calculating travel times for {quartier}...")
        
        # Reisezeiten für verschiedene Transportmittel berechnen
        for ziel_name, ziel_adresse in zielorte.items():
            for mode in ['transit', 'driving']:
                # Update progress
                processed += 1
                if processed % 10 == 0:
                    print(f"Progress: {processed}/{total_calculations} ({processed/total_calculations*100:.1f}%)")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
                # Reisezeit berechnen
                duration = get_travel_time(origin, ziel_adresse, mode)
                
                if duration is not None:
                    travel_times.append({
                        'Quartier': quartier,
                        'Zielort': ziel_name,
                        'Transportmittel': mode,
                        'Reisezeit_Minuten': round(duration, 1)
                    })
                else:
                    print(f"Could not calculate travel time from {quartier} to {ziel_name}.")

    # DataFrame erstellen
    df_travel_times = pd.DataFrame(travel_times)

    # Check if we got any travel times
    if df_travel_times.empty:
        print("ERROR: No travel times were calculated. Check your API key and network connection.")
        sys.exit(1)

    # Save the results
    travel_times_path = os.path.join(processed_dir, 'travel_times.csv')
    df_travel_times.to_csv(travel_times_path, index=False)
    print(f"Travel times saved to: {travel_times_path}")

    # Calculate average travel times per neighborhood
    df_avg_times = df_travel_times.groupby(['Quartier', 'Transportmittel']).agg({
        'Reisezeit_Minuten': ['mean', 'min', 'max']
    }).reset_index()

    df_avg_times.columns = ['Quartier', 'Transportmittel', 'Durchschnitt_Minuten', 'Min_Minuten', 'Max_Minuten']

    # Save average travel times
    avg_travel_times_path = os.path.join(processed_dir, 'avg_travel_times.csv')
    df_avg_times.to_csv(avg_travel_times_path, index=False)
    print(f"Average travel times saved to: {avg_travel_times_path}")
    
    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Total neighborhoods processed: {len(df_travel_times['Quartier'].unique())}")
    print(f"Total travel time calculations: {len(df_travel_times)}")
    print(f"Average travel times (transit): {df_travel_times[df_travel_times['Transportmittel'] == 'transit']['Reisezeit_Minuten'].mean():.1f} minutes")
    print(f"Average travel times (driving): {df_travel_times[df_travel_times['Transportmittel'] == 'driving']['Reisezeit_Minuten'].mean():.1f} minutes")
    
    print("\nTravel time generation completed successfully!")