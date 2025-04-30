"""
Reisezeit-Sammler für Zürich Immobilienpreis-Vorhersage

Dieses Skript sammelt Reisezeiten von verschiedenen Quartieren in Zürich
zu wichtigen Zielen wie Hauptbahnhof, ETH, Flughafen und Bahnhofstrasse.
Die Daten werden für die Immobilienpreis-Vorhersage verwendet.

Hinweis: Für echte Google Maps API-Daten wird ein API-Schlüssel benötigt.
Ohne Schlüssel werden realistische Platzhalter-Daten generiert.

Verwendung:
    1. Optional: Google Maps API-Schlüssel in einer .env Datei definieren:
       GOOGLE_MAPS_API_KEY=your_api_key_here
    2. Skript ausführen: python reisezeit_sammler.py
"""

import pandas as pd
import requests
import time
import json
import os
import random
from datetime import datetime

# Hauptfunktion am Ende hinzufügen
if __name__ == "__main__":
    main()

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

# Prüfen, ob der API-Schlüssel vorhanden ist
if GOOGLE_MAPS_API_KEY:
    print("Google Maps API-Schlüssel gefunden.")
else:
    print("Kein Google Maps API-Schlüssel gefunden. Es werden Platzhalter-Daten generiert.")

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
        # Realistischere Platzhalter-Zeiten je nach Ziel und Quartier
        def generate_realistic_travel_time(quartier, ziel):
            # Quartiere nahe am Zentrum
            zentrale_quartiere = ['Altstadt', 'City', 'Enge', 'Seefeld', 'Lindenhof', 'Niederdorf', 'Rathaus']
            # Quartiere weiter draußen
            aussere_quartiere = ['Affoltern', 'Oerlikon', 'Seebach', 'Schwamendingen', 'Altstetten', 'Albisrieden', 'Leimbach']
            
            base_time = 0
            
            if any(zentral.lower() in quartier.lower() for zentral in zentrale_quartiere):
                # Zentrale Quartiere haben kurze Reisezeiten
                base_time = 10
            elif any(aussen.lower() in quartier.lower() for aussen in aussere_quartiere):
                # Äußere Quartiere haben längere Reisezeiten
                base_time = 25
            else:
                # Mittlere Quartiere
                base_time = 15
            
            # Variiere basierend auf dem Ziel
            if 'Hauptbahnhof' in ziel:
                return max(5, base_time + random.randint(-5, 5))
            elif 'ETH' in ziel:
                return max(5, base_time + random.randint(-3, 7))
            elif 'Flughafen' in ziel:
                return max(15, base_time + 15 + random.randint(-5, 10))
            elif 'Bahnhofstrasse' in ziel:
                return max(5, base_time + random.randint(-5, 5))
            else:
                return max(5, base_time + random.randint(-5, 10))
        
        zeit = generate_realistic_travel_time(start, ziel)
        print(f"  Platzhalter für {start} nach {ziel}: {zeit} Minuten")
        return zeit
    
    # Google Maps API URL
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Parameter für die API-Anfrage (Pendlerzeit um 8:30 Uhr)
    params = {
        "origin": start,
        "destination": ziel,
        "mode": modus,
        "key": GOOGLE_MAPS_API_KEY,
        "departure_time": "now",  # Aktuelle Zeit
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
            zeit = random.randint(10, 50)
            print(f"  Platzhalter wird verwendet: {zeit} Minuten")
            return zeit
    except Exception as e:
        print(f"  Fehler: {e}")
        zeit = random.randint(10, 50)
        print(f"  Platzhalter wird verwendet: {zeit} Minuten")
        return zeit  # Platzhalter im Fehlerfall

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
        
        # Neuen Eintrag erstellen, falls nicht vorhanden
        if quartier not in reisezeiten:
            reisezeiten[quartier] = {}
        
        # Reisezeiten für alle Ziele abfragen
        for ziel_name, ziel_adresse in zielorte.items():
            # Überspringe, wenn wir bereits Daten für dieses Ziel haben
            if ziel_name in reisezeiten[quartier]:
                print(f"  Überspringe: {ziel_name} (bereits vorhanden)")
                continue
                
            # Reisezeit abfragen
            reisezeit = get_reisezeit(quartier, ziel_adresse)
            reisezeiten[quartier][ziel_name] = reisezeit
            
            # Zwischenspeichern nach jedem Ziel
            with open(cache_file, "w") as f:
                json.dump(reisezeiten, f, indent=2)
            
            # Kurze Pause, um API-Limits zu vermeiden
            if GOOGLE_MAPS_API_KEY:
                time.sleep(1)
    
    # Ergebnis zurückgeben
    return reisezeiten

def main():
    # Überprüfe, ob die Datei mit den Quartieren existiert
    quartier_file = 'data/processed/quartier_liste.csv'
    
    if not os.path.exists(quartier_file):
        print(f"Quartierliste nicht gefunden: {quartier_file}")
        print("Suche nach alternativen Quellen...")
        
        # Alternative 1: Verarbeitete Daten
        if os.path.exists('data/processed/quartier_daten.csv'):
            print("Verwende 'quartier_daten.csv'...")
            quartiere_df = pd.read_csv('data/processed/quartier_daten.csv')
            quartiere = sorted(quartiere_df['Quartier'].unique())
        # Alternative 2: Rohdaten
        elif os.path.exists('bau515od5155.csv'):
            print("Verwende Rohdaten 'bau515od5155.csv'...")
            df = pd.read_csv('bau515od5155.csv')
            if 'RaumLang' in df.columns:
                quartiere = sorted(df['RaumLang'].unique())
            else:
                print("FEHLER: RaumLang-Spalte nicht gefunden in Rohdaten")
                exit(1)
        # Alternative 3: Daten im data/raw Verzeichnis
        elif os.path.exists('data/raw/bau515od5155.csv'):
            print("Verwende Rohdaten in 'data/raw/bau515od5155.csv'...")
            df = pd.read_csv('data/raw/bau515od5155.csv')
            if 'RaumLang' in df.columns:
                quartiere = sorted(df['RaumLang'].unique())
            else:
                print("FEHLER: RaumLang-Spalte nicht gefunden in Rohdaten")
                exit(1)
        else:
            print("FEHLER: Keine Datenquelle für Quartiere gefunden!")
            print("Fallback: Verwende vordefinierte Quartierliste für Zürich")
            quartiere = [
                "Enge", "Wollishofen", "Leimbach", "Adliswil", "Kilchberg", 
                "Rüschlikon", "Thalwil", "Oberrieden", "Horgen", "Affoltern", 
                "Oerlikon", "Seebach", "Schwamendingen", "Altstetten", "Albisrieden", 
                "City", "Lindenhof", "Rathaus", "Hochschulen", "Bellevue", "Seefeld"
            ]
    else:
        # Quartierliste laden
        quartier_df = pd.read_csv(quartier_file)
        quartiere = quartier_df['Quartier'].tolist()
    
    print(f"Gefunden: {len(quartiere)} Quartiere")
    print(f"Beispiele: {', '.join(quartiere[:5])}")
    
    # Reisezeiten sammeln
    print(f"\nBeginne mit der Sammlung von Reisezeiten für {len(quartiere)} Quartiere...")
    reisezeiten = reisezeiten_sammeln(quartiere)
    
    # Speichern als JSON
    output_file = 'reisezeiten.json'
    with open(output_file, "w") as f:
        json.dump(reisezeiten, f, indent=2)
    
    # Kopie im processed-Verzeichnis speichern
    processed_file = 'data/processed/reisezeiten.json'
    with open(processed_file, "w") as f:
        json.dump(reisezeiten, f, indent=2)
    
    # Einzelne JSON-Dateien für jedes Quartier
    os.makedirs('data/processed/reisezeiten', exist_ok=True)
    for quartier, zeiten in reisezeiten.items():
        with open(f"data/processed/reisezeiten/{quartier.replace('/', '_')}.json", "w") as f:
            json.dump(zeiten, f, indent=2)
    
    print(f"\n✅ Reisezeiten erfolgreich gesammelt und gespeichert!")
    print(f"Hauptdatei: {output_file}")
    print(f"Kopie: {processed_file}")
    print(f"Einzeldateien: data/processed/reisezeiten/")
    
    # Statistiken ausgeben
    min_zeit = min([zeit for q in reisezeiten.values() for zeit in q.values()])
    max_zeit = max([zeit for q in reisezeiten.values() for zeit in q.values()])
    summe = sum([zeit for q in reisezeiten.values() for zeit in q.values()])
    anzahl = sum([len(q) for q in reisezeiten.values()])
    durchschnitt = summe / anzahl if anzahl > 0 else 0
    
    print(f"\nStatistik:")
    print(f"  - Gesammelte Reisezeiten: {anzahl}")
    print(f"  - Minimale Reisezeit: {min_zeit} Minuten")
    print(f"  - Maximale Reisezeit: {max_zeit} Minuten")
    print(f"  - Durchschnittliche Reisezeit: {durchschnitt:.1f} Minuten")