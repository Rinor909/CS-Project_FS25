"""
Reisezeit-Erfassungsskript für das Immobilienpreisvorhersage-Projekt Zürich
Dieses Skript sammelt echte Reisezeiten von Zürcher Quartieren zu wichtigen Zielen
mittels der Google Maps Directions API.
"""

import pandas as pd  # Für Datenmanipulation und -analyse
import requests  # Für HTTP-Anfragen an die API
import time  # Für Verzögerungen zwischen API-Anfragen
import json  # Zum Lesen/Schreiben von JSON-Dateien
import os  # Für Dateisystemoperationen
import datetime  # Für Datums- und Zeitberechnung
from dotenv import load_dotenv  # Für sichere API-Schlüsselverwaltung (neu)

# Versuche, .env-Datei zu laden, falls vorhanden
load_dotenv()

# API-Schlüssel aus Umgebungsvariable laden oder Placeholder nutzen
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "HIER_IHREN_API_SCHLÜSSEL_EINFÜGEN")

# Dateipfade definieren für Konsistenz mit datenbereinigung.py
INPUT_QUARTIERS_FILE = 'data/processed/quartier_liste.csv'
OUTPUT_REISEZEITEN_FILE = 'data/processed/reisezeiten.json'
CACHE_REISEZEITEN_FILE = 'data/processed/reisezeiten_cache.json'

# Verzeichnis sicherstellen
os.makedirs('data/processed', exist_ok=True)

# Wichtige Zielorte in Zürich
# Die Werte sind die vollständigen Adressen für eine genauere Ortsbestimmung
zielorte = {
    "Hauptbahnhof": "Zürich HB, Zürich",
    "ETH Zürich": "ETH Zürich, Rämistrasse 101, 8092 Zürich",
    "Flughafen Zürich": "Flughafen Zürich, 8058 Zürich",
    "Bahnhofstrasse": "Bahnhofstrasse, 8001 Zürich",
    "Paradeplatz": "Paradeplatz, 8001 Zürich",
    "Sihlcity": "Sihlcity, Kalanderplatz 1, 8045 Zürich",
    "Bellevue": "Bellevue, 8008 Zürich",
    "UZH Irchel Campus": "Universität Zürich Irchel, Winterthurerstrasse 190, 8057 Zürich"
}

# Funktion zur Ermittlung der Reisezeit zwischen zwei Orten
def get_reisezeit(start, ziel, modus="transit"):
    """
    Ermittelt die Reisezeit zwischen zwei Orten mittels Google Maps API
    
    Parameter:
    start (str): Startort (Adresse oder Koordinaten)
    ziel (str): Zielort (Adresse oder Koordinaten)
    modus (str): Fortbewegungsmittel - "driving" (Auto), "transit" (ÖV), 
                "walking" (zu Fuss), "bicycling" (Fahrrad)
    
    Rückgabe:
    int: Reisezeit in Minuten oder None bei Fehler
    """
    # "Zürich" hinzufügen, um sicherzustellen, dass wir den richtigen Ort erhalten
    if "Zürich" not in start and "Zurich" not in start:
        start = f"{start}, Zürich, Schweiz"
    
    # Überprüfen des API-Schlüssels
    if GOOGLE_MAPS_API_KEY == "HIER_IHREN_API_SCHLÜSSEL_EINFÜGEN":
        print("WARNUNG: Kein gültiger API-Schlüssel gefunden.")
        print("Platzhalter-Reisezeit wird zurückgegeben.")
        import random
        return random.randint(5, 60)  # Platzhalter: 5-60 Minuten
    
    # URL für die API-Anfrage erstellen
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Timestamp für einen typischen Mittwochmorgen um 8:30 Uhr
    # Berechne den nächsten Mittwoch
    now = datetime.datetime.now()
    days_ahead = (2 - now.weekday()) % 7  # 2 steht für Mittwoch (0-6 ist Mo-So)
    if days_ahead == 0:  # Wenn heute Mittwoch ist, nehme nächste Woche
        days_ahead = 7
    next_wednesday = now + datetime.timedelta(days=days_ahead)
    departure_time = next_wednesday.replace(hour=8, minute=30, second=0).timestamp()
    
    # Parameter für die Anfrage
    params = {
        "origin": start,  # Startpunkt
        "destination": ziel,  # Zielpunkt
        "mode": modus,  # Fortbewegungsmittel
        "key": GOOGLE_MAPS_API_KEY,  # API-Schlüssel
        "departure_time": int(departure_time),  # Korrigiert: Unix Timestamp
        "traffic_model": "best_guess"  # Beste Schätzung der Verkehrsbedingungen
    }
    
    try:
        # API-Anfrage senden
        response = requests.get(url, params=params)
        daten = response.json()
        
        # Prüfen, ob die Anfrage erfolgreich war
        if daten["status"] == "OK":
            # Reisezeit aus der ersten Route extrahieren
            reisezeit_sekunden = daten["routes"][0]["legs"][0]["duration"]["value"]
            reisezeit_minuten = reisezeit_sekunden // 60  # Umrechnung in Minuten
            
            print(f"  Reisezeit von {start} nach {ziel}: {reisezeit_minuten} Minuten")
            return reisezeit_minuten
        else:
            # Fehlerbehandlung, wenn die API keinen Status "OK" zurückgibt
            print(f"  Fehler bei der Ermittlung der Reisezeit: {daten['status']}")
            # Bei OVER_QUERY_LIMIT eine längere Pause einlegen
            if daten["status"] == "OVER_QUERY_LIMIT":
                print("  API-Limit überschritten. Warte 60 Sekunden...")
                time.sleep(60)
            return None
    except Exception as e:
        # Ausnahmebehandlung für alle anderen Fehler
        print(f"  Ausnahme bei der Ermittlung der Reisezeit: {e}")
        return None

# Funktion zum Sammeln von Reisezeiten für alle Quartiere
def reisezeiten_sammeln(quartiere):
    """
    Sammelt Reisezeiten von jedem Quartier zu wichtigen Zielen
    
    Parameter:
    quartiere (list): Liste der Quartiernamen
    
    Rückgabe:
    dict: Wörterbuch mit Reisezeiten
    """
    # Wörterbuch zum Speichern der Ergebnisse
    reisezeiten = {}
    
    # Prüfen, ob wir zwischengespeicherte Ergebnisse haben
    if os.path.exists(CACHE_REISEZEITEN_FILE):
        try:
            # Laden der zwischengespeicherten Ergebnisse
            with open(CACHE_REISEZEITEN_FILE, "r") as f:
                reisezeiten = json.load(f)
            print(f"Geladen: {len(reisezeiten)} Quartiere aus dem Cache")
        except json.JSONDecodeError:
            print(f"Fehler beim Laden der Cache-Datei. Erstelle neuen Cache.")
    
    # Für jedes Quartier
    for i, quartier in enumerate(quartiere):
        # Überspringen, wenn wir dieses Quartier bereits verarbeitet haben
        if quartier in reisezeiten and all(ziel in reisezeiten[quartier] for ziel in zielorte):
            print(f"Überspringe bereits verarbeitetes Quartier: {quartier}")
            continue
            
        print(f"Verarbeite {i+1}/{len(quartiere)}: {quartier}")
        
        # Initialisieren, falls das Quartier noch nicht im Wörterbuch ist
        if quartier not in reisezeiten:
            reisezeiten[quartier] = {}
        
        # Für jedes Ziel
        for ziel_name, ziel_adresse in zielorte.items():
            # Überspringen, wenn wir dieses Ziel bereits verarbeitet haben
            if ziel_name in reisezeiten[quartier]:
                continue
                
            # Reisezeit ermitteln
            reisezeit = get_reisezeit(quartier, ziel_adresse)
            
            # Ergebnis speichern
            if reisezeit:
                reisezeiten[quartier][ziel_name] = reisezeit
            else:
                # Standardwert, wenn keine Reisezeit ermittelt werden konnte
                reisezeiten[quartier][ziel_name] = 30  
            
            # Kurze Pause, um API-Ratenbegrenzungen zu vermeiden
            time.sleep(1.5)
        
        # Regelmäßiges Speichern des Fortschritts
        with open(CACHE_REISEZEITEN_FILE, "w") as f:
            json.dump(reisezeiten, f, indent=2)
        
        print(f"  Fortschritt gespeichert: {i+1}/{len(quartiere)} Quartiere")
    
    return reisezeiten

# Hauptausführung
if __name__ == "__main__":
    print("Reisezeiterfassungsskript für Zürcher Immobilienpreisvorhersage")
    
    if GOOGLE_MAPS_API_KEY == "HIER_IHREN_API_SCHLÜSSEL_EINFÜGEN":
        print("\nWARNUNG: Kein API-Schlüssel konfiguriert!")
        print("Um einen echten Google Maps API-Schlüssel zu verwenden:")
        print("1. Erstellen Sie eine .env-Datei im Projektverzeichnis")
        print("2. Fügen Sie folgende Zeile hinzu: GOOGLE_MAPS_API_KEY=Ihr_API_Schlüssel")
        print("3. Installieren Sie python-dotenv: pip install python-dotenv")
        print("\nOhne API-Schlüssel werden Platzhalter-Reisezeiten generiert.\n")
    
    # Laden der Quartierdaten
    try:
        # Versuchen, die Datei mit Quartierpreisen zu laden
        if os.path.exists(INPUT_QUARTIERS_FILE):
            quartier_daten = pd.read_csv(INPUT_QUARTIERS_FILE)
            quartiere = quartier_daten['Quartier'].unique()
        else:
            # Alternative Datei versuchen, falls die erste nicht existiert
            alt_file = 'quartier_preise.csv'
            if os.path.exists(alt_file):
                quartier_daten = pd.read_csv(alt_file)
                quartiere = quartier_daten['Quartier'].unique()
            else:
                raise FileNotFoundError(f"Weder {INPUT_QUARTIERS_FILE} noch {alt_file} gefunden")
        
        print(f"Sammle Reisezeiten für {len(quartiere)} Quartiere...")
        reisezeiten = reisezeiten_sammeln(quartiere)
        
        # Speichern der endgültigen Ergebnisse
        with open(OUTPUT_REISEZEITEN_FILE, "w") as f:
            json.dump(reisezeiten, f, indent=2)  # Einrückung für bessere Lesbarkeit
        
        print("Reisezeiterfassung abgeschlossen!")
        print(f"Ergebnisse in {OUTPUT_REISEZEITEN_FILE} gespeichert")
        
    except Exception as e:
        print(f"Fehler: {e}")
        print(f"Stellen Sie sicher, dass die Datei '{INPUT_QUARTIERS_FILE}' existiert.")
        print("Führen Sie zuerst datenbereinigung.py aus, um diese Datei zu erstellen.")