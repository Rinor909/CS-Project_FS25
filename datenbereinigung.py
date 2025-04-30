"""
Datenbereinigungsskript für Zürich Immobilienpreis-Projekt
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import random

# Debug-Information: Wo sind wir und was ist hier?
print(f"Aktuelles Verzeichnis: {os.getcwd()}")
print("Dateien im Verzeichnis:")
for file in os.listdir('.'):
    print(f"  - {file}")

# Ordner erstellen falls nicht existieren
os.makedirs('data/processed', exist_ok=True)

# Absolute Pfad zur aktuellen Datei und Projektordner
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Skript-Verzeichnis: {current_dir}")

# Versuche verschiedene Möglichkeiten, die CSV-Dateien zu finden
possible_paths = [
    # Direkter Pfad
    os.path.join(current_dir, 'bau515od5155.csv'),
    # Ein Verzeichnis höher
    os.path.join(current_dir, '..', 'bau515od5155.csv'),
    # Relativer Pfad vom aktuellen Arbeitsverzeichnis
    'bau515od5155.csv',
    # Im data-Unterordner
    os.path.join(current_dir, 'data', 'bau515od5155.csv'),
    # Im raw-Unterordner
    os.path.join(current_dir, 'data', 'raw', 'bau515od5155.csv')
]

# Finde den ersten Pfad, der existiert
neighborhood_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        neighborhood_data_path = path
        print(f"Gefunden: {path}")
        break

if not neighborhood_data_path:
    print("FEHLER: Konnte bau515od5155.csv nicht finden!")
    print("Überprüfe, ob die Datei in einem dieser Pfade existiert:")
    for path in possible_paths:
        print(f"  - {path}")
    exit(1)

# Das gleiche für die zweite Datei
building_age_data_path = None
for path in possible_paths:
    path = path.replace('bau515od5155.csv', 'bau515od5156.csv')
    if os.path.exists(path):
        building_age_data_path = path
        print(f"Gefunden: {path}")
        break

if not building_age_data_path:
    print("FEHLER: Konnte bau515od5156.csv nicht finden!")
    exit(1)

# Daten laden
print("Lade Datensätze...")
try:
    neighborhood_data = pd.read_csv(neighborhood_data_path)
    building_age_data = pd.read_csv(building_age_data_path)
    print(f"Daten erfolgreich geladen mit {len(neighborhood_data)} und {len(building_age_data)} Zeilen")
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit(1)

# Nur die Spalten behalten, die wir brauchen
# Quartier-Daten bereinigen
neighborhood_clean = neighborhood_data[[
    'Stichtagdatjahr',  # Jahr
    'HASTWELang',       # Miete oder Eigentum
    'RaumLang',         # Quartier
    'AnzZimmerLevel2Lang_noDM',  # Anzahl Zimmer
    'HAMedianPreis'     # Medianpreis
]]

# Spalten umbenennen damit's einfacher zu verstehen ist
neighborhood_clean = neighborhood_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'RaumLang': 'Quartier',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'QuartierPreis'
})

# Das gleiche für Baualter-Daten
building_age_clean = building_age_data[[
    'Stichtagdatjahr',
    'HASTWELang',
    'BaualterLang_noDM',
    'AnzZimmerLevel2Lang_noDM',
    'HAMedianPreis'
]]

building_age_clean = building_age_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'BaualterLang_noDM': 'Baualter',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'BaualterPreis'
})

# Fehlende Werte mit Durchschnitt füllen
if neighborhood_clean['QuartierPreis'].isnull().sum() > 0:
    mean_price = neighborhood_clean['QuartierPreis'].mean()
    neighborhood_clean['QuartierPreis'] = neighborhood_clean['QuartierPreis'].fillna(mean_price)
    print(f"Fehlende Quartierpreise mit Durchschnitt gefüllt: {mean_price:.2f}")

if building_age_clean['BaualterPreis'].isnull().sum() > 0:
    mean_price = building_age_clean['BaualterPreis'].mean()
    building_age_clean['BaualterPreis'] = building_age_clean['BaualterPreis'].fillna(mean_price)
    print(f"Fehlende Baualterpreise mit Durchschnitt gefüllt: {mean_price:.2f}")

# Schlüssel erstellen für späteren Join
# Format: Jahr_Immobilientyp_Zimmeranzahl
neighborhood_clean['Schluessel'] = (
    neighborhood_clean['Jahr'].astype(str) + '_' +
    neighborhood_clean['Immobilientyp'] + '_' +
    neighborhood_clean['Zimmeranzahl']
)

building_age_clean['Schluessel'] = (
    building_age_clean['Jahr'].astype(str) + '_' +
    building_age_clean['Immobilientyp'] + '_' +
    building_age_clean['Zimmeranzahl']
)

# Eindeutige Werte finden
quartiere = neighborhood_clean['Quartier'].unique()
baualter = building_age_clean['Baualter'].unique()
schluessel = pd.concat([neighborhood_clean['Schluessel'], building_age_clean['Schluessel']]).unique()

print(f"Gefunden: {len(quartiere)} Quartiere, {len(baualter)} Baualter, {len(schluessel)} Kombinationen")

# Datensätze zusammenführen
# Liste für neue Daten
kombinierte_daten = []

# Für jeden Schlüssel
for key in schluessel:
    # Aufteilung der Komponenten
    jahr, typ, zimmer = key.split('_', 2)
    
    # Finde passende Quartierpreise
    quartier_preise = neighborhood_clean[neighborhood_clean['Schluessel'] == key]
    
    if len(quartier_preise) > 0:
        for _, quartier_row in quartier_preise.iterrows():
            quartier = quartier_row['Quartier']
            quartier_preis = quartier_row['QuartierPreis']
            
            # Finde passende Baualterspreise
            alter_preise = building_age_clean[building_age_clean['Schluessel'] == key]
            
            if len(alter_preise) > 0:
                for _, alter_row in alter_preise.iterrows():
                    alter = alter_row['Baualter']
                    alter_preis = alter_row['BaualterPreis']
                    
                    # Kombinierte Daten erstellen
                    # Einfacher Ansatz: Geschätzter Preis = Durchschnitt von Quartier- und Baualterspreis
                    kombinierte_daten.append({
                        'Jahr': int(jahr),
                        'Immobilientyp': typ,
                        'Zimmeranzahl': zimmer,
                        'Quartier': quartier,
                        'Baualter': alter,
                        'QuartierPreis': quartier_preis,
                        'BaualterPreis': alter_preis,
                        'GeschaetzterPreis': (quartier_preis + alter_preis) / 2
                    })

# In DataFrame umwandeln
kombinierter_df = pd.DataFrame(kombinierte_daten)
print(f"Kombinierter Datensatz mit {len(kombinierter_df)} Einträgen erstellt")

# Reisezeitdaten integrieren
# Schauen ob reisezeiten.json existiert
reisezeiten_pfade = [
    os.path.join(current_dir, 'data', 'processed', 'reisezeiten.json'),
    os.path.join(current_dir, 'reisezeiten.json'),
    'reisezeiten.json',
    os.path.join('data', 'processed', 'reisezeiten.json')
]

reisezeiten_vorhanden = False
reisezeiten = {}

# Suche nach der reisezeiten.json Datei
for pfad in reisezeiten_pfade:
    if os.path.exists(pfad):
        try:
            with open(pfad, "r") as f:
                reisezeiten = json.load(f)
            print(f"Reisezeitdaten aus {pfad} geladen für {len(reisezeiten)} Quartiere")
            reisezeiten_vorhanden = True
            break
        except Exception as e:
            print(f"Fehler beim Laden von {pfad}: {e}")

if not reisezeiten_vorhanden:
    print("Keine Reisezeitdaten gefunden - erstelle Platzhalter")

# Wichtige Ziele in Zürich
ziele = [
    'Hauptbahnhof', 
    'ETH Zürich', 
    'Flughafen Zürich', 
    'Bahnhofstrasse',
    'Paradeplatz',
    'Sihlcity',
    'Bellevue',
    'UZH Irchel Campus'
]

# Reisezeiten zum Datensatz hinzufügen
for ziel in ziele:
    spaltenname = f'Reisezeit_{ziel}'
    
    if reisezeiten_vorhanden:
        # Echte Reisezeiten verwenden
        kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
            # Wenn keine Zeit für ein Quartier/Ziel gefunden wird, 30 Min. als Standardwert
            lambda quartier: reisezeiten.get(quartier, {}).get(ziel, 30)
        )
    else:
        # Platzhalter erstellen - zwischen 5 und 60 Minuten zufällig
        for quartier in quartiere:
            if quartier not in reisezeiten:
                reisezeiten[quartier] = {}
            if ziel not in reisezeiten[quartier]:
                # Geringe Reisezeit für Zentrumsquartiere und länger für periphere
                # Hier als Beispiel einfach zufällig
                reisezeiten[quartier][ziel] = random.randint(5, 60)
        
        # Platzhalter zum Datensatz hinzufügen
        kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
            lambda quartier: reisezeiten[quartier][ziel]
        )

# Daten speichern
print("Speichere bereinigte Daten...")
os.makedirs('data/processed', exist_ok=True)

neighborhood_clean.to_csv('data/processed/quartier_daten.csv', index=False)
building_age_clean.to_csv('data/processed/baualter_daten.csv', index=False)
kombinierter_df.to_csv('data/processed/zuerich_immobilien_komplett.csv', index=False)

# Reisezeitdaten als Platzhalter speichern
if not reisezeiten_vorhanden:
    with open('data/processed/reisezeiten_platzhalter.json', 'w') as f:
        json.dump(reisezeiten, f, indent=2)

# Quartierliste für Reisezeitsammlung speichern
pd.DataFrame({'Quartier': quartiere}).to_csv('data/processed/quartier_liste.csv', index=False)

print(f"Fertig! {len(quartiere)} Quartiere und {len(kombinierter_df)} kombinierte Einträge gespeichert.")