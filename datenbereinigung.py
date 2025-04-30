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
import numpy as np

# Debugging-Information: Wo sind wir?
print(f"Aktuelles Verzeichnis: {os.getcwd()}")
print("Dateien im Verzeichnis:")
for file in os.listdir('.'):
    print(f"  - {file}")

# Ordner erstellen falls nicht existieren
os.makedirs('data/processed', exist_ok=True)

print("Immobilienpreis-Analyse Zürich: Datenbereinigung")
print("=" * 50)

print("Suche nach Datensätzen...")

# Prüfen, ob die Dateien im aktuellen Verzeichnis oder im data/raw Verzeichnis existieren
file1 = 'bau515od5155.csv'
file2 = 'bau515od5156.csv'
data_paths = ['.', 'data/raw']

neighborhood_data_path = None
building_age_data_path = None

for path in data_paths:
    if os.path.exists(os.path.join(path, file1)):
        neighborhood_data_path = os.path.join(path, file1)
    if os.path.exists(os.path.join(path, file2)):
        building_age_data_path = os.path.join(path, file2)

if neighborhood_data_path and building_age_data_path:
    print(f"Dateien gefunden!")
    print(f"Neighborhood data: {neighborhood_data_path}")
    print(f"Building age data: {building_age_data_path}")
else:
    print("FEHLER: Konnte die CSV-Dateien nicht finden!")
    print("Bitte stellen Sie sicher, dass die folgenden Dateien existieren:")
    print(f"- {file1}")
    print(f"- {file2}")
    exit(1)

# Daten laden
print("Lade Datensätze...")
try:
    df1 = pd.read_csv(neighborhood_data_path)
    df2 = pd.read_csv(building_age_data_path)
    print(f"Daten erfolgreich geladen mit {len(df1)} und {len(df2)} Zeilen")
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit(1)

# Daten erkunden
print("\nVerfügbare Spalten in Neighborhoods-Datensatz:")
for col in df1.columns:
    print(f"  - {col}")

print("\nVerfügbare Spalten in Building Age-Datensatz:")
for col in df2.columns:
    print(f"  - {col}")

# Quartier-Daten bereinigen
print("\nBereinige Quartier-Daten...")
neighborhood_clean = df1[[
    'Stichtagdatjahr',  # Jahr
    'HASTWELang',       # Miete oder Eigentum
    'RaumLang',         # Quartier
    'AnzZimmerLevel2Lang_noDM',  # Anzahl Zimmer
    'HAMedianPreis'     # Medianpreis
]].copy()

# Überprüfen auf NaN-Werte
print(f"NaN-Werte in Quartier-Daten vor Bereinigung: {neighborhood_clean.isna().sum().sum()}")

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
building_age_clean = df2[[
    'Stichtagdatjahr',
    'HASTWELang',
    'BaualterLang_noDM',
    'AnzZimmerLevel2Lang_noDM',
    'HAMedianPreis'
]].copy()

# Überprüfen auf NaN-Werte
print(f"NaN-Werte in Baualter-Daten vor Bereinigung: {building_age_clean.isna().sum().sum()}")

building_age_clean = building_age_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'BaualterLang_noDM': 'Baualter',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'BaualterPreis'
})

# Fehlende Werte identifizieren und mit Durchschnitt füllen
nan_count_neighborhood = neighborhood_clean['QuartierPreis'].isna().sum()
if nan_count_neighborhood > 0:
    mean_price = neighborhood_clean['QuartierPreis'].mean()
    neighborhood_clean['QuartierPreis'] = neighborhood_clean['QuartierPreis'].fillna(mean_price)
    print(f"Fehlende Quartierpreise mit Durchschnitt gefüllt: {mean_price:.2f} ({nan_count_neighborhood} Werte)")

nan_count_building = building_age_clean['BaualterPreis'].isna().sum()
if nan_count_building > 0:
    mean_price = building_age_clean['BaualterPreis'].mean()
    building_age_clean['BaualterPreis'] = building_age_clean['BaualterPreis'].fillna(mean_price)
    print(f"Fehlende Baualterpreise mit Durchschnitt gefüllt: {mean_price:.2f} ({nan_count_building} Werte)")

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
quartiere = sorted(neighborhood_clean['Quartier'].unique())
baualter = sorted(building_age_clean['Baualter'].unique())
schluessel = pd.concat([neighborhood_clean['Schluessel'], building_age_clean['Schluessel']]).unique()

print(f"Gefunden: {len(quartiere)} Quartiere, {len(baualter)} Baualter, {len(schluessel)} Kombinationen")

# Datensätze zusammenführen
print("Kombiniere Datensätze...")
kombinierte_daten = []

# Für jeden Schlüssel
for key in schluessel:
    # Aufteilung der Komponenten
    key_parts = key.split('_', 2)
    if len(key_parts) != 3:
        print(f"Warnung: Ungültiger Schlüssel ignoriert: {key}")
        continue
        
    jahr, typ, zimmer = key_parts
    
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

# Prüfen ob wir Daten haben
if len(kombinierter_df) == 0:
    print("WARNUNG: Keine kombinierten Daten gefunden!")
    # Fallback: Erstelle ein einfaches kartesisches Produkt
    print("Erstelle vereinfachte Daten als Fallback...")
    
    # Erstelle kartesisches Produkt der letzten Jahre
    neueste_jahre = sorted(neighborhood_clean['Jahr'].unique())[-3:]
    fallback_daten = []
    
    for jahr in neueste_jahre:
        for quartier in quartiere:
            quartier_daten = neighborhood_clean[
                (neighborhood_clean['Jahr'] == jahr) & 
                (neighborhood_clean['Quartier'] == quartier)
            ]
            
            if len(quartier_daten) == 0:
                continue
                
            # Finde Durchschnittspreise für dieses Quartier
            quartier_preis = quartier_daten['QuartierPreis'].mean()
            
            for alter in baualter:
                alter_daten = building_age_clean[
                    (building_age_clean['Jahr'] == jahr) & 
                    (building_age_clean['Baualter'] == alter)
                ]
                
                if len(alter_daten) == 0:
                    continue
                    
                alter_preis = alter_daten['BaualterPreis'].mean()
                
                # Standard-Zimmeranzahl
                zimmeranzahlen = ['1 Zimmer', '2 Zimmer', '3 Zimmer', '4 Zimmer', '5+ Zimmer']
                for zimmer in zimmeranzahlen:
                    fallback_daten.append({
                        'Jahr': jahr,
                        'Immobilientyp': 'Eigentumswohnung',
                        'Zimmeranzahl': zimmer,
                        'Quartier': quartier,
                        'Baualter': alter,
                        'QuartierPreis': quartier_preis,
                        'BaualterPreis': alter_preis,
                        'GeschaetzterPreis': (quartier_preis + alter_preis) / 2
                    })
    
    kombinierter_df = pd.DataFrame(fallback_daten)

print(f"Kombinierter Datensatz mit {len(kombinierter_df)} Einträgen erstellt")

# Reisezeitdaten integrieren oder Platzhalter erstellen
reisezeiten_pfad = 'reisezeiten.json'
reisezeiten_vorhanden = os.path.exists(reisezeiten_pfad)

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

# Definiere eine Funktion für realistischere Reisezeiten basierend auf Quartieren
def generate_realistic_travel_time(quartier, ziel):
    # Quartiere nahe am Zentrum
    zentrale_quartiere = ['Altstadt', 'City', 'Enge', 'Seefeld', 'Lindenhof', 'Niederdorf', 'Rathaus']
    # Quartiere weiter draußen
    aussere_quartiere = ['Affoltern', 'Oerlikon', 'Seebach', 'Schwamendingen', 'Altstetten', 'Albisrieden', 'Leimbach']
    
    base_time = 0
    
    if any(zentral in quartier for zentral in zentrale_quartiere):
        # Zentrale Quartiere haben kurze Reisezeiten
        base_time = 10
    elif any(aussen in quartier for aussen in aussere_quartiere):
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

# Wenn keine Reisezeitdaten vorhanden sind, erstelle Platzhalter
if not reisezeiten_vorhanden:
    print("Erstelle Platzhalter für Reisezeiten...")
    for quartier in quartiere:
        reisezeiten[quartier] = {}
        for ziel in ziele:
            # Realistischere Reisezeiten basierend auf Quartier und Ziel
            reisezeiten[quartier][ziel] = generate_realistic_travel_time(quartier, ziel)

# Reisezeiten zum Datensatz hinzufügen
for ziel in ziele:
    spaltenname = f'Reisezeit_{ziel}'
    kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
        lambda quartier: reisezeiten.get(quartier, {}).get(ziel, 30)
    )

# Daten speichern
print("Speichere bereinigte Daten...")

# Hole den neuesten Stand der Daten
neuestes_jahr = max(kombinierter_df['Jahr'])
print(f"Neuestes Jahr in den Daten: {neuestes_jahr}")

# Erstelle eine Stichprobe der neuesten Daten für schnelleres Laden in Streamlit
neueste_daten = kombinierter_df[kombinierter_df['Jahr'] == neuestes_jahr]
sample_size = min(5000, len(neueste_daten))
stichprobe = neueste_daten.sample(n=sample_size) if sample_size > 0 else neueste_daten

# Spezielle Stichprobe für Streamlit Demo
demo_stichprobe = stichprobe.sample(n=min(100, len(stichprobe))) if len(stichprobe) > 0 else stichprobe

# Speichere alle Daten
neighborhood_clean.to_csv('data/processed/quartier_daten.csv', index=False)
building_age_clean.to_csv('data/processed/baualter_daten.csv', index=False)
kombinierter_df.to_csv('data/processed/zuerich_immobilien_komplett.csv', index=False)
stichprobe.to_csv('data/processed/zuerich_immobilien_stichprobe.csv', index=False)
demo_stichprobe.to_csv('data/processed/zuerich_immobilien_demo.csv', index=False)

# Quartierliste für Reisezeitsammlung speichern
pd.DataFrame({'Quartier': quartiere}).to_csv('data/processed/quartier_liste.csv', index=False)

# Speichere einfache Statistiken für Streamlit-App
statistiken = {
    'quartiere': len(quartiere),
    'baualter': list(baualter),
    'zimmeranzahlen': sorted(kombinierter_df['Zimmeranzahl'].unique().tolist()),
    'preis_min': float(kombinierter_df['GeschaetzterPreis'].min()),
    'preis_max': float(kombinierter_df['GeschaetzterPreis'].max()),
    'preis_avg': float(kombinierter_df['GeschaetzterPreis'].mean()),
    'preis_median': float(kombinierter_df['GeschaetzterPreis'].median()),
    'neuestes_jahr': int(neuestes_jahr)
}

with open('data/processed/statistiken.json', 'w') as f:
    json.dump(statistiken, f, indent=2)

# Wenn keine echten Reisezeitdaten vorhanden sind, speichere die Platzhalter
if not reisezeiten_vorhanden:
    with open('data/processed/reisezeiten_platzhalter.json', 'w') as f:
        json.dump(reisezeiten, f, indent=2)
    # Kopie im Hauptverzeichnis für einfachen Zugriff
    with open('reisezeiten.json', 'w') as f:
        json.dump(reisezeiten, f, indent=2)

print("\n✅ Datenbereinigung abgeschlossen!")
print(f"Ergebnis: {len(quartiere)} Quartiere und {len(kombinierter_df)} kombinierte Einträge")
print(f"Stichprobe mit {len(stichprobe)} Einträgen für schnelleres Laden erstellt")
print("\nBereinigte Daten gespeichert in: data/processed/")
print("Für echte Reisezeitdaten führen Sie als nächstes 'reisezeit_sammler.py' aus")