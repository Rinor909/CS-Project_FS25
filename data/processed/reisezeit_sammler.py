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

# Ersetzen Sie dies durch Ihren tatsächlichen API-Schlüssel
# Um einen Schlüssel zu erhalten:
# 1. Besuchen Sie die Google Cloud Console
# 2. Erstellen Sie ein Projekt
# 3. Aktivieren Sie die Directions API
# 4. Erstellen Sie einen API-Schlüssel unter "Anmeldedaten"
GOOGLE_MAPS_API_KEY = "HIER_IHREN_API_SCHLÜSSEL_EINFÜGEN"

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
    
    # URL für die API-Anfrage erstellen
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Parameter für die Anfrage
    params = {
        "origin": start,  # Startpunkt
        "destination": ziel,  # Zielpunkt
        "mode": modus,  # Fortbewegungsmittel
        "key": GOOGLE_MAPS_API_KEY,  # API-Schlüssel
        # Ergebnisse für die Pendlerzeit morgens (8:30 Uhr an einem Mittwoch)
        "departure_time": "next_wednesday+08:30:00",
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
    if os.path.exists("reisezeiten_cache.json"):
        # Laden der zwischengespeicherten Ergebnisse
        with open("reisezeiten_cache.json", "r") as f:
            reisezeiten = json.load(f)
        print(f"Geladen: {len(reisezeiten)} Quartiere aus dem Cache")
    
    # Für jedes Quartier
    for i, quartier in enumerate(quartiere):
        # Überspringen, wenn wir dieses Quartier bereits verarbeitet haben
        if quartier in reisezeiten:
            print(f"Überspringe bereits verarbeitetes Quartier: {quartier}")
            continue
            
        print(f"Verarbeite {i+1}/{len(quartiere)}: {quartier}")
        reisezeiten[quartier] = {}
        
        # Für jedes Ziel
        for ziel_name, ziel_adresse in zielorte.items():
            # Reisezeit ermitteln
            reisezeit = get_reisezeit(quartier, ziel_adresse)
            
            # Ergebnis speichern
            if reisezeit:
                reisezeiten[quartier][ziel_name] = reisezeit
            else:
                # Standardwert, wenn keine Reisezeit ermittelt werden konnte
                reisezeiten[quartier][ziel_name] = 30  
            
            # Kurze Pause, um API-Ratenbegrenzungen zu vermeiden
            time.sleep(1)
        
        # Regelmäßiges Speichern des Fortschritts
        with open("reisezeiten_cache.json", "w") as f:
            json.dump(reisezeiten, f)
        
        print(f"  Fortschritt gespeichert: {i+1}/{len(quartiere)} Quartiere")
    
    return reisezeiten

# Hauptausführung
if __name__ == "__main__":
    # Laden der Quartierdaten
    try:
        # Versuchen, die Datei mit Quartierpreisen zu laden
        quartier_daten = pd.read_csv('quartier_preise.csv')
        quartiere = quartier_daten['Quartier'].unique()
        
        print(f"Sammle Reisezeiten für {len(quartiere)} Quartiere...")
        reisezeiten = reisezeiten_sammeln(quartiere)
        
        # Speichern der endgültigen Ergebnisse
        with open("reisezeiten.json", "w") as f:
            json.dump(reisezeiten, f, indent=2)  # Einrückung für bessere Lesbarkeit
        
        print("Reisezeiterfassung abgeschlossen!")
        print("Ergebnisse in reisezeiten.json gespeichert")
        
    except Exception as e:
        print(f"Fehler: {e}")
        print("Stellen Sie sicher, dass die Datei 'quartier_preise.csv' existiert.")
        print("Führen Sie zuerst datenbereinigung_mit_reisezeiten.py aus, um diese Datei zu erstellen.")