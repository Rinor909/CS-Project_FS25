"""
Reisezeit-Sammler für Zürich Immobilienpreis-Vorhersage

Dieses Skript sammelt Reisezeiten von verschiedenen Quartieren in Zürich
zu wichtigen Zielen wie Hauptbahnhof, ETH, Flughafen und Bahnhofstrasse.
Die Daten werden für die Immobilienpreis-Vorhersage verwendet.

Hinweis: Für echte Google Maps API-Daten wird ein API-Schlüssel benötigt.
Ohne Schlüssel werden realistische Platzhalter-Daten generiert.
"""

import pandas as pd
import requests
import time
import json
import os
import random

# Debug-Information: Wo sind wir?
print(f"Aktuelles Verzeichnis: {os.getcwd()}")
print("Dateien im Verzeichnis:")
for file in os.listdir('.'):
    print(f"  - {file}")

# Versuche dotenv zu importieren für API-Schlüssel
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env-Datei laden falls vorhanden
    print("dotenv erfolgreich geladen.")
except ImportError:
    print("Info: python-dotenv nicht installiert. API-Schlüssel direkt eingeben oder als Variable setzen.")

# API-Schlüssel aus Umgebungsvariable oder direkt eingeben
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Wichtige Orte in Zürich
zielorte = {
    "Hauptbahnhof": "Zürich HB, Zürich",
    "ETH Zürich": "ETH Zürich, Rämistrasse 101, 8092 Zürich",
    "Flughafen Zürich": "Flughafen Zürich, 8058 Zürich",
    "Bahnhofstrasse": "Bahnhofstrasse, 8001 Zürich"
}

def get_reisezeit(start, ziel, modus="transit"):
    """Ermittelt die Reisezeit zwischen zwei Orten mit Google Maps API oder als Platzhalter"""
    
    # Sicherstellen, dass wir in Zürich suchen
    if "Zürich" not in start and "Zurich" not in start:
        start = f"{start}, Zürich, Schweiz"
    
    # Wenn kein API-Schlüssel, Platzhalter-Daten zurückgeben
    if not GOOGLE_MAPS_API_KEY:
        print(f"  Platzhalter für {start} nach {ziel}: ", end="")
        
        # Realistischere Platzhalter-Zeiten je nach Ziel
        if "Flughafen" in ziel:
            # Flughafen ist weiter weg
            zeit = random.randint(20, 60)
        elif "Hauptbahnhof" in ziel or "Bahnhofstrasse" in ziel:
            # Zentrale Orte sind tendenziell näher
            zeit = random.randint(5, 40)
        else:
            zeit = random.randint(10, 50)
        
        print(f"{zeit} Minuten (Platzhalter)")
        return zeit
    
    # Google Maps API URL
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Parameter für die API-Anfrage (Pendlerzeit um 8:30 Uhr)
    params = {
        "origin": start,
        "destination": ziel,
        "mode": modus,
        "key": GOOGLE_MAPS_API_KEY,
        "departure_time": 1620202200,  # Feste Zeit für Konsistenz
        "traffic_model": "best_guess"
    }
    
    try:
        # API-Anfrage senden
        response = requests.get(url, params=params)
        data = response.json()
        
        # Überprüfen, ob die Anfrage erfolgreich war
        if data["status"] == "OK":
            # Reisezeit aus der ersten Route extrahieren
            reisezeit_sekunden = data["routes"][0]["legs"][0]["duration"]["value"]
            reisezeit_minuten = reisezeit_sekunden // 60  # Umrechnung in Minuten
            
            print(f"  Reisezeit von {start} nach {ziel}: {reisezeit_minuten} Minuten")
            return reisezeit_minuten
        else:
            print(f"  Fehler bei API-Anfrage: {data['status']}")
            
            # Wenn API-Limit erreicht, länger warten
            if data["status"] == "OVER_QUERY_LIMIT":
                print("  API-Limit erreicht! Warte 60 Sekunden...")
                time.sleep(60)
            
            # Platzhalter zurückgeben
            return random.randint(10, 50)
    except Exception as e:
        print(f"  Fehler: {e}")
        return random.randint(10, 50)  # Platzhalter im Fehlerfall

def reisezeiten_sammeln(quartiere):
    """Sammelt Reisezeiten für alle Quartiere zu allen Zielen"""
    
    # Ergebnis-Dictionary
    reisezeiten = {}
    
    # Prüfen, ob wir bereits Daten im Cache haben
    cache_file = 'data/processed/reisezeiten_cache.json'
    
    # Cache-Verzeichnis erstellen falls nicht vorhanden
    os.makedirs('data/processed', exist_ok=True)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                reisezeiten = json.load(f)
            print(f"Cache geladen: {len(reisezeiten)} Quartiere bereits verarbeitet")
        except json.JSONDecodeError:
            print("Cache-Datei beschädigt. Starte neu.")
    
    # Schleife durch alle Quartiere
    for i, quartier in enumerate(quartiere):
        print(f"[{i+1}/{len(quartiere)}] Verarbeite: {quartier}")
        
        # Überspringen, wenn dieses Quartier bereits komplett ist
        if quartier in reisezeiten and all(ziel in reisezeiten[quartier] for ziel in zielorte):
            print(f"  Überspringe: {quartier} (bereits verarbeitet)")
            continue