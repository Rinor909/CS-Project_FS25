"""
Datenbereinigungsskript für das Immobilienpreisvorhersage-Projekt Zürich

Dieses Skript bereinigt die Datensätze und führt sie zusammen.

"""

import pandas as pd  # Bibliothek für Datenanalyse und -manipulation
import matplotlib.pyplot as plt  # Bibliothek für Visualisierungen
import json  # Bibliothek zum Lesen/Schreiben von JSON-Dateien
import os  # Bibliothek für Dateisystemoperationen
import random  # Bibliothek für Zufallszahlen (für Platzhalter)

# Sicherstellen, dass Verzeichnisse existieren
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Dateipfade definieren
NEIGHBORHOOD_FILE = 'bau515od5155.csv'
BUILDING_AGE_FILE = 'bau515od5156.csv'
REISEZEITEN_FILE = 'data/processed/reisezeiten.json'  # Geänderter Pfad für Konsistenz

# Schritt 1 : Laden der CSV-Dateien in pandas DataFrame-Objekte
try:
    neighborhood_data = pd.read_csv(NEIGHBORHOOD_FILE) # Preise nach Quartier
    building_age_data = pd.read_csv(BUILDING_AGE_FILE) # Preise nach Baualter
    print(f"Datensätze erfolgreich geladen: {len(neighborhood_data)} Quartiereinträge, {len(building_age_data)} Baualtereinträge")
except FileNotFoundError as e:
    print(f"Fehler beim Laden der Datensätze: {e}")
    print("Bitte stellen Sie sicher, dass die CSV-Dateien im richtigen Verzeichnis liegen.")
    exit(1)

# Schritt 2 : Bereinigung der Datensätze
# Wir behalten nur die wichtigsten Spalten für unsere Analyse
neighborhood_clean = neighborhood_data[[
    'Stichtagdatjahr', # Jahr
    'HASTWELang', # Typ (Miete/Eigentum)
    'RaumLang', # Quartier
    'AnzZimmerLevel2Lang_noDM', # Anzahl Zimmer
    'HAMedianPreis' # Medianpreis
]]

# Wir umbennen die Spalten für bessere Lesbarkeit
neighborhood_clean = neighborhood_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'RaumLang': 'Quartier',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'QuartierPreis'
})

# Das Gleiche für die Baualtersdaten
building_age_clean = building_age_data[[
    'Stichtagdatjahr',           # Jahr
    'HASTWELang',                # Typ (Miete/Eigentum)
    'BaualterLang_noDM',         # Baualter
    'AnzZimmerLevel2Lang_noDM',  # Anzahl Zimmer
    'HAMedianPreis'              # Medianpreis
]]

# Wir umbennen die Spalten für bessere Lesbarkeit
building_age_clean = building_age_clean.rename(columns={
    'Stichtagdatjahr': 'Jahr',
    'HASTWELang': 'Immobilientyp',
    'BaualterLang_noDM': 'Baualter',
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',
    'HAMedianPreis': 'BaualterPreis'
})

# Schritt 3: Behandlung von fehlenden Werten
# Wir überprüfen, ob der Preis fehlende Werte hat und füllen diese mit dem Durchschnitt
if neighborhood_clean['QuartierPreis'].isnull().sum() > 0:
    mean_price = neighborhood_clean['QuartierPreis'].mean()
    neighborhood_clean['QuartierPreis'] = neighborhood_clean['QuartierPreis'].fillna(mean_price)
    print(f"Fehlende Quartierpreise mit Durchschnitt gefüllt: {mean_price:.2f}")

# Das gleiche für Baualter Datensatz
if building_age_clean['BaualterPreis'].isnull().sum() > 0:
    mean_price = building_age_clean['BaualterPreis'].mean()
    building_age_clean['BaualterPreis'] = building_age_clean['BaualterPreis'].fillna(mean_price)
    print(f"Fehlende Baualterpreise mit Durchschnitt gefüllt: {mean_price:.2f}")

# Schritt 4 : Erstellen eines gemeinsamen Schlüssels für beide Datensätze
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

# Schritt 5: Zusammenführen der Datensätze
# Eindeutige Werte für die Kombinationen ermitteln
quartiere = neighborhood_clean['Quartier'].unique()
baualter = building_age_clean['Baualter'].unique()
schluessel = pd.concat([neighborhood_clean['Schluessel'], building_age_clean['Schluessel']]).unique()

print(f"Gefundene eindeutige Werte: {len(quartiere)} Quartiere, {len(baualter)} Baualter, {len(schluessel)} Schlüssel")

# Leere Liste für den kombinierten Datensatz
kombinierte_daten = []

# Für jeden Schlüssel (Jahr, Immobilientyp, Zimmeranzahl)
for key in schluessel:
    # Jahr, Immobilientyp und Zimmeranzahl extrahieren
    jahr, typ, zimmer = key.split('_', 2)

    # Preisdaten für alle Quartiere mit diesem Schlüssel finden
    quartier_preise = neighborhood_clean[neighborhood_clean['Schluessel'] == key]
    # Wenn es Quartierdaten für diesen Schlüssel gibt
    if len(quartier_preise) > 0:
        for _, quartier_row in quartier_preise.iterrows():
            quartier = quartier_row['Quartier']
            quartier_preis = quartier_row['QuartierPreis']
            
            # Alle passenden Baualtersdaten für diesen Schlüssel finden
            alter_preise = building_age_clean[building_age_clean['Schluessel'] == key]
            
            # Wenn es Baualtersdaten für diesen Schlüssel gibt
            if len(alter_preise) > 0:
                for _, alter_row in alter_preise.iterrows():
                    alter = alter_row['Baualter']
                    alter_preis = alter_row['BaualterPreis']
                    
                    # Kombinierten Datensatz erstellen
                    kombinierte_daten.append({
                        'Jahr': int(jahr),
                        'Immobilientyp': typ,
                        'Zimmeranzahl': zimmer,
                        'Quartier': quartier,
                        'Baualter': alter,
                        'QuartierPreis': quartier_preis,
                        'BaualterPreis': alter_preis,
                        'GeschaetzterPreis': (quartier_preis + alter_preis) / 2  # Einfacher Durchschnitt
                    })

# In DataFrame umwandeln
kombinierter_df = pd.DataFrame(kombinierte_daten)
print(f"Kombinierter Datensatz erstellt mit {len(kombinierter_df)} Einträgen")

# Schritt 6: Reisezeitdaten integrieren (falls vorhanden oder mit Platzhaltern)
# Prüfen, ob eine reisezeiten.json Datei existiert
if os.path.exists(REISEZEITEN_FILE):
    # Reisezeiten aus Datei laden
    with open(REISEZEITEN_FILE, "r") as f:
        reisezeiten = json.load(f)
    
    print(f"Reisezeitdaten für {len(reisezeiten)} Quartiere geladen")
    reisezeiten_vorhanden = True
else:
    print(f"Keine Reisezeitdaten unter {REISEZEITEN_FILE} gefunden. Es werden Platzhalter erstellt.")
    reisezeiten_vorhanden = False
    # Platzhalter erstellen
    reisezeiten = {}

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

# Reisezeiten zum kombinierten Datensatz hinzufügen
for ziel in ziele:
    spaltenname = f'Reisezeit_{ziel}'
    
    if reisezeiten_vorhanden:
        # Echte Reisezeiten verwenden
        kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
            lambda quartier: reisezeiten.get(quartier, {}).get(ziel, 30)  # Standardwert 30 Minuten
        )
    else:
        # Platzhalter für jedes Quartier erstellen
        for quartier in quartiere:
            if quartier not in reisezeiten:
                reisezeiten[quartier] = {}
            if ziel not in reisezeiten[quartier]:
                reisezeiten[quartier][ziel] = random.randint(5, 60)
        
        # Platzhalter zum Datensatz hinzufügen
        kombinierter_df[spaltenname] = kombinierter_df['Quartier'].apply(
            lambda quartier: reisezeiten[quartier][ziel]
        )

# Schritt 7: Daten speichern
# Bereinigte Einzeldatensätze speichern
neighborhood_clean.to_csv('data/processed/quartier_daten.csv', index=False)
building_age_clean.to_csv('data/processed/baualter_daten.csv', index=False)

# Zusammengeführten Datensatz speichern (Hauptergebnis)
kombinierter_df.to_csv('data/processed/zuerich_immobilien_komplett.csv', index=False)

# Auch die Reisezeitdaten als JSON speichern (falls es nur Platzhalter waren)
if not reisezeiten_vorhanden:
    with open('data/processed/reisezeiten_platzhalter.json', 'w') as f:
        json.dump(reisezeiten, f, indent=2)

# Save a CSV with just Quartier names for travel time collection
quartier_daten = pd.DataFrame({'Quartier': quartiere})
quartier_daten.to_csv('data/processed/quartier_liste.csv', index=False)

print("Datensatzbereinigung abgeschlossen. Alle Dateien wurden in 'data/processed/' gespeichert.")
print(f"- {len(neighborhood_clean)} Quartierpreise gespeichert")
print(f"- {len(building_age_clean)} Baualterpreise gespeichert")
print(f"- {len(kombinierter_df)} kombinierte Einträge gespeichert")
print(f"- {len(quartiere)} Quartiere für Reisezeiterfassung gespeichert")