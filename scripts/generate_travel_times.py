import pandas as pd
import numpy as np
import os
import json
import time
import requests
import sys

# Um den API-Schlüssel zu lesen, musste ich eine secrets.toml-Datei auf GitHub verwenden, da dies auf normalem Weg nicht funktionierte
# GitHub meldete, dass mein API-Schlüssel offengelegt wurde, was ein Sicherheitsrisiko darstellte; ich war mit dieser Methode zum Lesen des API-Schlüssels nicht vertraut
# Dieser Teil von Zeile 13 bis 43 wurde mithilfe von KI (ChatGPT) erstellt
# Funktion zum direkten Lesen des API-Schlüssels von GitHub
def get_api_key_from_github():
    secrets_url = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/.streamlit/secrets.toml' # URL zur Rohdatei secrets.toml im GitHub-Repository

    try:
        response = requests.get(secrets_url) # Versuch, die secrets-Datei herunterzuladen
        if response.status_code == 200: # Prüfen, ob die Anfrage erfolgreich war (HTTP 200 Antwort)
            content = response.text # Den Inhalt der Datei abrufen
            for line in content.split('\n'): # Die Datei Zeile für Zeile durchsuchen, um den Schlüssel zu finden
                if line.startswith('GOOGLE_MAPS_API_KEY'): # Suche nach der Zeile, die den Google Maps API-Schlüssel definiert
                    api_key = line.split('=')[1].strip().strip('"').strip("'") # Den Wert extrahieren
                    return api_key 
        else:
            print(f"Failed to get secrets file: HTTP {response.status_code}") # HTTP-Anfrage fehlgeschlagen
    except Exception as e:
        print(f"Error getting API key from GitHub: {e}") # HTTP-Anfrage fehlgeschlagen
    return None
# HTTP-Anfrage fehlgeschlagen   
print("Loading API key from GitHub...")
GOOGLE_MAPS_API_KEY = get_api_key_from_github() # Erster Versuch, den API-Schlüssel von GitHub abzurufen
# Rückgriff auf Umgebungsvariable, falls GitHub fehlschlägt
if not GOOGLE_MAPS_API_KEY: # Zweiter Versuch, falls das Abrufen des API-Schlüssels von GitHub fehlschlug
    print("Trying environment variable...")
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") # Suche nach dem API-Schlüssel in Umgebungsvariablen
# Prüfen, ob ein API-Schlüssel vorhanden ist
if not GOOGLE_MAPS_API_KEY: # Wenn beide Methoden fehlschlagen, Fehlermeldung anzeigen
    print("ERROR: No Google Maps API key found.")
    print("Please make sure your API key is correctly set in:")
    print("https://github.com/Rinor909/zurich-real-estate/blob/main/.streamlit/secrets.toml")
    sys.exit(1) # Beende das Programm, da der API-Schlüssel für die Hauptfunktionalität erforderlich ist
else:
    print(f"Successfully loaded API key: {GOOGLE_MAPS_API_KEY[:5]}...{GOOGLE_MAPS_API_KEY[-5:]}") # Erfolgsmeldung mit maskiertem API-Schlüssel, um ein unbeabsichtigtes Offenlegen des vollständigen Schlüssels in Logs zu verhindern

# Wir verwenden erneut direkte GitHub-URLs, um unsere Daten zu laden
url_quartier = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/quartier_processed.csv'
df_quartier = pd.read_csv(url_quartier) # read the CSV file from a URL into a panda DataFrame
quartiere = df_quartier['Quartier'].unique() # extracts a unique list of neighborhood names from the 'Quartier' column

# Lokales Ausgabeverzeichnis zum Speichern definieren; anschließend wird das Ergebnis auf GitHub hochgeladen, um es im weiteren Code zu verwenden
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data" 
processed_dir = os.path.join(output_dir, "processed") 
os.makedirs(processed_dir, exist_ok=True)

# Wichtige Zielorte in Zürich
# Jede Destination wird für API-Abfragen einer Adresse zugeordnet
zielorte = {
    'Hauptbahnhof': 'Zürich Hauptbahnhof, Zürich, Schweiz',
    'ETH': 'ETH Zürich, Rämistrasse 101, 8092 Zürich, Schweiz',
    'Flughafen': 'Flughafen Zürich, Kloten, Schweiz',
    'Bahnhofstrasse': 'Bahnhofstrasse, Zürich, Schweiz'
}

# Tatsächliche zentrale Koordinaten für jedes Quartier
# Diese Koordinaten stellen den ungefähren Mittelpunkt jeder Region dar
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
# Behandlung von Stadtteilen, die sich im Datensatz befinden, aber nicht im Koordinaten-Dictionary enthalten sind, durch Zuweisung eines Standardwertes (Zentrum)
missing_quartiers = []
for quartier in quartiere:
    if quartier not in quartier_koordinaten:
        missing_quartiers.append(quartier)
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
    # Pfad zur Cache-Datei
    # Ich musste diese Funktionalität zum Caching der API-Ergebnisse hinzufügen, da ich nur eine begrenzte Anzahl an API-Anfragen im kostenlosen Plan durchführen konnte
    # Da ich mit Caching nicht vertraut war, wurde KI verwendet, um den Code für die Zeilen 142 bis 158 zu erstellen
    cache_file = os.path.join(processed_dir, 'travel_time_cache.json') # Wir definieren einen Cache-Dateipfad zum Speichern vorheriger API-Ergebnisse
    # Cache laden, falls vorhanden
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            # Beschädigte Cache-Datei behandeln
            print(f"Warning: Corrupted cache file. Creating new cache.")
            cache = {}
    else:
        cache = {}
    # Cache-Key erstellen
    cache_key = f"{origin['lat']},{origin['lng']}_TO_{destination}_{mode}"
    # Wenn Ergebnis im Cache, aus Cache zurückgeben
    if cache_key in cache:
        return cache[cache_key]
    
    url = "https://maps.googleapis.com/maps/api/directions/json" # URL für Google Maps Directions API
    params = {     # Startkoordinaten
        'origin': f"{origin['lat']},{origin['lng']}", # Start coordinates
        'destination': destination, # Zieladresse
        'mode': mode, # Transportmodus
        'key': GOOGLE_MAPS_API_KEY, # API-Authentifizierung
        'departure_time': 'now', # Aktuelle Zeit verwenden
    }
    
    # Anfrage senden
    # Anfrage an die Google Maps API senden und Antwort verarbeiten
    response = requests.get(url, params=params)
    data = response.json()
    # Prüfen, ob die Anfrage erfolgreich war
    if data['status'] == 'OK':
        # Reisezeit aus der ersten Route extrahieren
        route = data['routes'][0]
        leg = route['legs'][0]
        duration_seconds = leg['duration']['value']
        duration_minutes = duration_seconds / 60
        # Im Cache speichern # AI was used to code lines 186 to 188
        cache[cache_key] = duration_minutes
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        return duration_minutes
    return None

if __name__ == "__main__": # Liste für Reisedaten initialisieren
    # DataFrame für Reisezeiten
    travel_times = []
    # Schneller API-Test, um zu überprüfen, ob die API-Verbindung funktioniert, bevor alle Berechnungen gestartet werden
    if get_travel_time(quartier_koordinaten.get(quartiere[0]), zielorte['Hauptbahnhof'], 'transit') is None:
        sys.exit(1)
    # Reisezeit für jeden Stadtteil zu allen Zielorten berechnen
    for quartier in quartiere: # process each neighborhood
        origin = quartier_koordinaten.get(quartier) # get coordinates for the current neighborhood
        if not origin:
            continue     
        for ziel_name, ziel_adresse in zielorte.items(): # calculate travel times to all destinations
            for mode in ['transit', 'driving']: # Für verschiedene Transportmodi berechnen
                # Kleine Verzögerung hinzufügen, um Rate-Limiting der Google Maps API zu vermeiden
                time.sleep(0.1)
                # Reisezeit berechnen
                duration = get_travel_time(origin, ziel_adresse, mode)
                if duration is not None: # Erfolgreiche Berechnungsergebnisse speichern
                    travel_times.append({
                        'Quartier': quartier, # Starting neighborhood
                        'Zielort': ziel_name, # Destination name
                        'Transportmittel': mode, # Transportation method
                        'Reisezeit_Minuten': round(duration, 1) # Travel time rounded to one decimal
                    })

    # Gesammelte Reisezeitdaten in ein DataFrame umwandeln, um die Analyse zu erleichtern
    df_travel_times = pd.DataFrame(travel_times)
    # Gesamten Datensatz der Reisezeiten in CSV speichern, um ihn später weiterzuverwenden (similar to what we previously did, i.e. saving locally and uploading on GitHub)
    travel_times_path = os.path.join(processed_dir, 'travel_times.csv')
    df_travel_times.to_csv(travel_times_path, index=False)
    # Durchschnittliche, minimale und maximale Reisezeiten pro Stadtteil berechnen
    df_avg_times = df_travel_times.groupby(['Quartier', 'Transportmittel']).agg({
        'Reisezeit_Minuten': ['mean', 'min', 'max'] # Key stats for analysis
    }).reset_index()
    # Spalten für die Ausgabe umbenennen
    df_avg_times.columns = ['Quartier', 'Transportmittel', 'Durchschnitt_Minuten', 'Min_Minuten', 'Max_Minuten']
    # Zusammenfassende Statistiken zur schnellen Referenz und Visualisierung speichern
    avg_travel_times_path = os.path.join(processed_dir, 'avg_travel_times.csv')
    df_avg_times.to_csv(avg_travel_times_path, index=False)
