"""
Datenbereinigungsskript für Zürich Immobilienpreis-Projekt

Dieses Skript bereitet die Daten für die Immobilienpreis-Analyse vor:
1. Laden und Bereinigen der Quartier-Daten
2. Laden und Bereinigen der Baualter-Daten
3. Kombinieren beider Datensätze für die Preisvorhersage
"""

import pandas as pd
import os
import json
import random

# Ordner erstellen falls nicht existieren
os.makedirs('data/processed', exist_ok=True)

print("Immobilienpreis-Analyse Zürich: Datenbereinigung")
print("=" * 50)

# Pfade zu den CSV-Dateien definieren (Die Dateien müssen im gleichen Verzeichnis sein)
print("Suche nach Datensätzen...")

# Mögliche Pfade zu den CSV-Dateien
possible_paths = [
    'bau515od5155.csv',               # Im aktuellen Verzeichnis
    '../bau515od5155.csv',            # Ein Verzeichnis höher
    '../../bau515od5155.csv',         # Zwei Verzeichnisse höher
    './data/raw/bau515od5155.csv',    # Im data/raw Unterverzeichnis
    '../data/raw/bau515od5155.csv',   # Im data/raw Unterverzeichnis eine Ebene höher
]

# Finde den ersten Pfad, der existiert
neighborhood_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        neighborhood_data_path = path
        print(f"Gefunden: {path}")
        break

# Gleiches für die zweite Datei
building_age_data_path = None
for path in possible_paths:
    building_path = path.replace('bau515od5155.csv', 'bau515od5156.csv')
    if os.path.exists(building_path):
        building_age_data_path = building_path
        print(f"Gefunden: {building_path}")
        break

if not neighborhood_data_path or not building_age_data_path:
    print("FEHLER: Konnte die CSV-Dateien nicht finden!")
    print("Bitte stellen Sie sicher, dass die folgenden Dateien existieren:")
    print("- bau515od5155.csv")
    print("- bau515od5156.csv")
    print("\nMögliche Speicherorte:")
    for path in possible_paths:
        print(f"- {path}")
    exit(1)

# Daten laden
print("Lade Datensätze...")
try:
    neighborhood_df = pd.read_csv(neighborhood_data_path)
    building_age_df = pd.read_csv(building_age_data_path)
    print(f"Daten erfolgreich geladen mit {len(neighborhood_df)} und {len(building_age_df)} Zeilen")
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit(1)

# Quartier-Daten bereinigen
print("Bereinige Quartier-Daten...")
neighborhood_clean = neighborhood_df[[
    'Stichtagdatjahr',  # Jahr
    'HASTWELang',       # Miete oder Eigentum
    'RaumLang',         # Quartier
    'AnzZimmerLevel2Lang_noDM',  # Anzahl Zimmer
    'HAMedianPreis'     # Medianpreis
]]

# Spalten umbenennen
neighborhood_clean = neighborhood_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'RaumLang': 'Quartier',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'QuartierPreis'
})

# Baualter-Daten bereinigen
print("Bereinige Baualter-Daten...")
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
print("Kombiniere Datensätze...")
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
                    # Geschätzter Preis = Durchschnitt von Quartier- und Baualterspreis
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

# Reisezeitdaten integrieren oder Platzhalter erstellen
reisezeiten_pfade = [
    'reisezeiten.json',
    'data/processed/reisezeiten.json',
    '../reisezeiten.json'
]

reisezeiten_pfad = None
reisezeiten_vorhanden = False

for pfad in reisezeiten_pfade:
    if os.path.exists(pfad):
        reisezeiten_pfad = pfad
        reisezeiten_vorhanden = True
        break

if reisezeiten_vorhanden:
    print(f"Lade Reisezeitdaten aus {reisezeiten_pfad}...")
    with open(reisezeiten_pfad, 'r') as f:
        reisezeiten = json.load(f)
else:
    print("Keine Reisezeitdaten gefunden - erstelle Platzhalter")
    # Erstelle ein leeres Dictionary für Reisezeiten
    reisezeiten = {}

# Wichtige Ziele in Zürich
ziele = [
    'Hauptbahnhof', 
    'ETH Zürich', 
    'Flughafen Zürich', 
    'Bahnhofstrasse'
]

# Wenn keine Reisezeitdaten vorhanden sind, erstelle Platzhalter
if not reisezeiten_vorhanden:
    print("Erstelle Platzhalter für Reisezeiten...")
    for quartier in quartiere:
        reisezeiten[quartier] = {}
        for ziel in ziele:
            # Platzhalter-Reisezeiten zwischen 5 und 60 Minuten
            reisezeiten[quartier][ziel] = random.randint(5, 60)

# Reisezeiten zum Datensatz hinzufügen
for ziel in ziele:
    spaltenname = f'Reisezeit_{ziel}'
    kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
        lambda quartier: reisezeiten.get(quartier, {}).get(ziel, 30)
    )

# Daten speichern
print("Speichere bereinigte Daten...")

neighborhood_clean.to_csv('data/processed/quartier_daten.csv', index=False)
building_age_clean.to_csv('data/processed/baualter_daten.csv', index=False)
kombinierter_df.to_csv('data/processed/zuerich_immobilien_komplett.csv', index=False)

# Quartierliste für Reisezeitsammlung speichern
pd.DataFrame({'Quartier': quartiere}).to_csv('data/processed/quartier_liste.csv', index=False)

# Wenn keine echten Reisezeitdaten vorhanden sind, speichere die Platzhalter
if not reisezeiten_vorhanden:
    with open('data/processed/reisezeiten_platzhalter.json', 'w') as f:
        json.dump(reisezeiten, f, indent=2)

print("\n✅ Datenbereinigung abgeschlossen!")
print(f"Ergebnis: {len(quartiere)} Quartiere und {len(kombinierter_df)} kombinierte Einträge")
print("\nFür echte Reisezeitdaten führen Sie als nächstes 'reisezeit_sammler.py' aus")
print("Bereinigte Daten gespeichert in: data/processed/")