import pandas as pd
import numpy as np
import os

# Hier gebe ich ein lokales Ausgabeverzeichnis an, da ich nicht in der Lage war, 
# sie direkt auf GitHub zu speichern (ich habe 2 Tage lang versucht, dies zu beheben, ohne Erfolg).
# Ich speichere also die Exporte lokal und lade sie dann auf GitHub hoch, um sie später für die anderen Skripte zu verwenden.
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed")

# Ich stelle sicher, dass die Datenverzeichnisse existieren oder erstelle sie falls nicht
# processed_dir: zum Speichern bereinigter/transformierter CSV-Datendateien
# models-Verzeichnis: zum Speichern trainierter Machine-Learning-Modelldateienos.makedirs(processed_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

# Die CSV-Dateien werden direkt von den Raw-URLs unseres GitHub-Repositories geladen,
# da der lokale Dateizugriff auf verschiedenen Systemen Probleme verursachte
url_quartier = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/raw/bau515od5155.csv'
url_baualter = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/raw/bau515od5156.csv'
df_quartier = pd.read_csv(url_quartier)
df_baualter = pd.read_csv(url_baualter)

# Wichtige Spalten aus dem Quartier-Datensatz auswählen und umbenennen
# Diese Extraktion verbessert die Lesbarkeit und vereinfacht die Weiterverarbeitung
quartier_spalten = {
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'RaumLang': 'Quartier',                 # Name des Stadtquartiers 
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_quartier_clean = df_quartier[quartier_spalten.keys()].copy()
df_quartier_clean.rename(columns=quartier_spalten, inplace=True)

# Ähnliche Transformation für den Baualter-Datensatz durchführen
baualter_spalten = {
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'BaualterLang_noDM': 'Baualter',        # Baualtersklasse als Text
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_baualter_clean = df_baualter[baualter_spalten.keys()].copy()
df_baualter_clean.rename(columns=baualter_spalten, inplace=True)

# Fehlende Werte durch kontextabhängige Mediane oder Mittelwerte ersetzen
# Dies erhöht die Datenqualität und verhindert Verzerrungen durch fehlende Werte
for df in [df_quartier_clean, df_baualter_clean]:
    if 'Quartier' in df.columns:
        # Für Quartier-Datensatz: Gruppierung nach Quartier und Zimmeranzahl
        # MedianPreis: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Quartier', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
        
        # PreisProQm: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Quartier', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
    else:
        # Für Baualter-Datensatz: Gruppierung nach Baualter und Zimmeranzahl
        # MedianPreis: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Baualter', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
        
        # PreisProQm: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Baualter', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
    
    # Globale Mediane für verbleibende fehlende Werte verwenden
    median_price = df['MedianPreis'].median()
    df['MedianPreis'].fillna(0 if pd.isna(median_price) else median_price, inplace=True)
    
    median_price_per_sqm = df['PreisProQm'].median()
    df['PreisProQm'].fillna(0 if pd.isna(median_price_per_sqm) else median_price_per_sqm, inplace=True)

# Feature-Engineering: Textbasierte Zimmeranzahl in numerischen Wert konvertieren
# Dies ermöglicht mathematische Operationen und ML-Modellierung
def zimmer_zu_int(zimmer_str):
    """Extrahiert die Zimmeranzahl aus dem String-Format"""
    try:
        return int(zimmer_str.split('-')[0]) # Extrahiert '3' aus '3-Zimmer'
    except:
        return np.nan

df_quartier_clean['Zimmeranzahl_num'] = df_quartier_clean['Zimmeranzahl'].apply(zimmer_zu_int)
df_baualter_clean['Zimmeranzahl_num'] = df_baualter_clean['Zimmeranzahl'].apply(zimmer_zu_int)

# Feature-Engineering: Baualter-Kategorien in geschätztes Baujahr umwandeln
# Wandelt kategorische Variable in kontinuierlichen Wert für ML-Modellierung um
def baualter_zu_jahr(baualter_str):
    """Wandelt Baualter-Text in ungefähres Baujahr um"""
    try:
        # Zeitspannen in einen Mittelwert umwandeln (z.B. '1981-2000' → 1990.5)
        if '-' in baualter_str:
            jahre = baualter_str.split('-')
            return (int(jahre[0]) + int(jahre[1])) / 2
        # Spezielle Kategorien interpretieren
        elif 'vor' in baualter_str:
            return 1919 # Standardwert für 'vor 1919'
        # Format "nach 2015" oder "seit 2015"
        elif 'nach' in baualter_str or 'seit' in baualter_str:
            return 2015 # Standardwert für neuere Gebäude
        else:
            return np.nan
    except:
        return np.nan

if 'Baualter' in df_baualter_clean.columns:
    df_baualter_clean['Baujahr'] = df_baualter_clean['Baualter'].apply(baualter_zu_jahr)

# Gebäudealter-Features zum Quartier-Datensatz hinzufügen
# Für jedes Quartier, Jahr und Zimmeranzahl den Durchschnittspreis pro Baualtersklasse ermitteln
# und dann mit dem Quartier-Datensatz zusammenführen

# Aggregieren der Baualter-Daten nach Jahr und Zimmeranzahl
df_baualter_agg = df_baualter_clean.groupby(['Jahr', 'Zimmeranzahl_num']).agg({
    'MedianPreis': 'mean', # Durchschnittspreis pro Baualter-Kategorie
    'Baujahr': 'mean'   # Durchschnittliches Baujahr
}).reset_index()

# Spalten für bessere Lesbarkeit umbenennen
df_baualter_agg.rename(columns={
    'MedianPreis': 'MedianPreis_Baualter',
    'Baujahr': 'Durchschnitt_Baujahr'
}, inplace=True)

# Quartier- und Baualter-Daten mittels Jahr und Zimmeranzahl zusammenführen
# Dadurch werden beide Dimensionen in einem gemeinsamen Datensatz vereint
df_merged = pd.merge(
    df_quartier_clean,
    df_baualter_agg,
    on=['Jahr', 'Zimmeranzahl_num'],
    how='left'
)

# Feature-Engineering: Preisverhältnis zwischen Quartier und Baualter berechnen
# Dieses Feature zeigt die relative Preispositionierung eines Quartiers im Vergleich zum Baujahrstandard
df_merged['Preis_Verhältnis'] = df_merged['MedianPreis'] / df_merged['MedianPreis_Baualter']

# Nur die neuesten Daten für das Modelltraining filtern
# Aktuelle Daten sind relevanter für Preisvorhersagen
neuestes_jahr = df_merged['Jahr'].max()
df_final = df_merged[df_merged['Jahr'] == neuestes_jahr].copy()

# Feature-Engineering: Relatives Preisniveau pro Quartier berechnen
# Ermöglicht Vergleich zwischen Quartieren unabhängig von absoluten Preisen
quartier_avg_preis = df_final.groupby('Quartier')['MedianPreis'].mean()
gesamtpreis_avg = quartier_avg_preis.mean()
quartier_preisniveau = (quartier_avg_preis / gesamtpreis_avg).to_dict()

df_final['Quartier_Preisniveau'] = df_final['Quartier'].map(quartier_preisniveau)

# Zeilen mit fehlenden Werten in Kernvariablen entfernen
# Sichert die Datenqualität für das ML-Modell
df_final.dropna(subset=['MedianPreis', 'Quartier', 'Zimmeranzahl_num'], inplace=True)

# Verbleibende fehlende Werte durch Mediane ersetzen
# Sicherstellen, dass das Dataset vollständig ist, ohne Informationsverlust
for column in df_final.columns:
    if df_final[column].dtype in [np.float64, np.int64]:
        # Robuste Behandlung von Spalten, die nur NaN-Werte enthalten könnten
        df_final = df_final.copy()
        median_value = df_final[column].median()
        if pd.isna(median_value):
            # Fallback auf 0 wenn kein Median berechnet werden kann
            df_final.loc[:, column] = df_final[column].fillna(0)
        else:
            # Standardfall: Mit Median füllen
            df_final.loc[:, column] = df_final[column].fillna(median_value)

# Quartiere für ML-Modelle in numerische Codes konvertieren
# One-Hot-Encoding-Vorbereitung für kategorische Variablen
df_final['Quartier_Code'] = pd.Categorical(df_final['Quartier']).codes

# Aufbereitete Datensätze in CSV-Dateien speichern
# Diese Dateien werden für Modelltraining und App-Visualisierung verwendet
quartier_path = os.path.join(processed_dir, 'quartier_processed.csv')
baualter_path = os.path.join(processed_dir, 'baualter_processed.csv')
final_path = os.path.join(processed_dir, 'modell_input_final.csv')

df_quartier_clean.to_csv(quartier_path, index=False)
df_baualter_clean.to_csv(baualter_path, index=False)
df_final.to_csv(final_path, index=False)