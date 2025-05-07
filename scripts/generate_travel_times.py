import pandas as pd
import numpy as np
import os
import json
import time
import requests
import streamlit as st

# Use Streamlit secrets for the API key
try:
    GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
except Exception as e:
    print(f"Error loading API key from Streamlit secrets: {e}")
    GOOGLE_MAPS_API_KEY = None  # Will use simulations if the key is not found

# Lade Quartier-Daten
try:
    df_quartier = pd.read_csv('data/processed/quartier_processed.csv')
except FileNotFoundError:
    # Graceful handling if file doesn't exist yet
    print("Quartier data not found. Run data_preparation.py first.")
    # Create a directory structure if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    # Create an empty dataframe with the expected structure
    df_quartier = pd.DataFrame(columns=['Quartier'])
    quartiere = []
else:
    # Liste der eindeutigen Quartiere
    quartiere = df_quartier['Quartier'].unique()

# Wichtige Zielorte in Zürich
zielorte = {
    'Hauptbahnhof': 'Zürich Hauptbahnhof, Zürich, Schweiz',
    'ETH': 'ETH Zürich, Rämistrasse 101, 8092 Zürich, Schweiz',
    'Flughafen': 'Flughafen Zürich, Kloten, Schweiz',
    'Bahnhofstrasse': 'Bahnhofstrasse, Zürich, Schweiz'
}

# Fiktive zentrale Koordinaten für jedes Quartier (normalerweise würde man diese aus GeoJSON-Daten beziehen)
# Dies ist eine Vereinfachung. Reale Implementierung würde präzisere Daten verwenden.
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
    'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841}
}

# Für fehlende Quartiere Standardwerte hinzufügen
for quartier in quartiere:
    if quartier not in quartier_koordinaten:
        # Setze Default-Koordinaten für Zürich Zentrum
        quartier_koordinaten[quartier] = {'lat': 47.3769, 'lng': 8.5417}

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
    # Ensure the processed directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    cache_file = f'data/processed/travel_time_cache.json'
    
    # Cache laden, falls vorhanden
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted cache file
            cache = {}
    else:
        cache = {}
    
    # Cache-Key erstellen
    cache_key = f"{origin['lat']},{origin['lng']}_TO_{destination}_{mode}"
    
    # Wenn Ergebnis im Cache, aus Cache zurückgeben
    if cache_key in cache:
        return cache[cache_key]
    
    # Wenn kein API-Key verfügbar, simulierte Reisezeit zurückgeben
    if not GOOGLE_MAPS_API_KEY:
        # Zufällige aber realistische Reisezeit simulieren
        # Hier könnte man später eine bessere Simulation basierend auf Entfernung implementieren
        simulated_time = np.random.normal(30, 10)  # Mittelwert 30 Min, Standardabweichung 10 Min
        simulated_time = max(5, min(90, simulated_time))  # Zwischen 5 und 90 Minuten begrenzen
        
        # Im Cache speichern
        cache[cache_key] = simulated_time
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
            
        return simulated_time
    
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
            
            # Fallback to simulation if API fails
            simulated_time = np.random.normal(30, 10)
            simulated_time = max(5, min(90, simulated_time))
            return simulated_time
    except Exception as e:
        print(f"Fehler bei der API-Anfrage: {e}")
        # Fallback to simulation
        simulated_time = np.random.normal(30, 10)
        simulated_time = max(5, min(90, simulated_time))
        return simulated_time

if __name__ == "__main__":
    # Only execute this part when script is run directly
    # DataFrame für Reisezeiten erstellen
    travel_times = []

    # Für jedes Quartier die Reisezeiten zu allen Zielorten berechnen
    for quartier in quartiere:
        origin = quartier_koordinaten.get(quartier)
        if not origin:
            print(f"Keine Koordinaten für {quartier} gefunden.")
            continue
            
        print(f"Berechne Reisezeiten für {quartier}...")
        
        # Reisezeiten für verschiedene Transportmittel berechnen
        for ziel_name, ziel_adresse in zielorte.items():
            for mode in ['transit', 'driving']:
                time.sleep(0.2)  # Kurze Pause, um API-Limits einzuhalten
                
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
                    print(f"Konnte Reisezeit von {quartier} nach {ziel_name} nicht berechnen.")

    # DataFrame erstellen
    df_travel_times = pd.DataFrame(travel_times)

    # Ergebnisse speichern
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
        
    df_travel_times.to_csv('data/processed/travel_times.csv', index=False)

    print(f"Reisezeiten wurden in 'data/processed/travel_times.csv' gespeichert.")

    # Durchschnittliche Reisezeiten pro Quartier berechnen
    if not df_travel_times.empty:
        df_avg_times = df_travel_times.groupby(['Quartier', 'Transportmittel']).agg({
            'Reisezeit_Minuten': ['mean', 'min', 'max']
        }).reset_index()

        df_avg_times.columns = ['Quartier', 'Transportmittel', 'Durchschnitt_Minuten', 'Min_Minuten', 'Max_Minuten']

        # Durchschnittliche Reisezeitdaten speichern
        df_avg_times.to_csv('data/processed/avg_travel_times.csv', index=False)

        print(f"Durchschnittliche Reisezeiten wurden in 'data/processed/avg_travel_times.csv' gespeichert.")
    else:
        print("Keine Reisezeiten berechnet. Bitte überprüfen Sie Ihre Daten und API-Konfiguration.")