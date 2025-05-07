import pandas as pd
import numpy as np
import os


df_quartier = pd.read_csv('Rinor909/zurich-real-estate/bau515od5155.csv', sep=',')
df_baualter = pd.read_csv('Rinor909/zurich-real-estate/bau515od5156.csv', sep=',')

# Datenbereinigung - Quartier-Datensatz
# Nur die wichtigsten Spalten behalten
quartier_spalten = {
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'RaumLang': 'Quartier',                 # Name des Stadtquartiers 
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_quartier_clean = df_quartier[quartier_spalten.keys()].copy()
df_quartier_clean.rename(columns=quartier_spalten, inplace=True)

# Datenbereinigung - Baualter-Datensatz
baualter_spalten = {
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'BaualterLang_noDM': 'Baualter',        # Baualtersklasse als Text
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_baualter_clean = df_baualter[baualter_spalten.keys()].copy()
df_baualter_clean.rename(columns=baualter_spalten, inplace=True)

# Fehlende Werte behandeln
# Strategie: Medianwerte nach Gruppierung verwenden
for df in [df_quartier_clean, df_baualter_clean]:
    if 'Quartier' in df.columns:
        # MedianPreis: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Quartier', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean()))
        
        # PreisProQm: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Quartier', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean()))
    else:
        # MedianPreis: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Baualter', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean()))
        
        # PreisProQm: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Baualter', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean()))
    
    # Verbleibende fehlende Werte durch allgemeine Mediane ersetzen
    df['MedianPreis'].fillna(df['MedianPreis'].median(), inplace=True)
    df['PreisProQm'].fillna(df['PreisProQm'].median(), inplace=True)

# Datentypen anpassen und Feature-Engineering
# Zimmeranzahl: Von Text (z.B. "2-Zimmer") zu Zahl (2) konvertieren
def zimmer_zu_int(zimmer_str):
    """Extrahiert die Zimmeranzahl aus dem String-Format"""
    try:
        return int(zimmer_str.split('-')[0])
    except:
        return np.nan

df_quartier_clean['Zimmeranzahl_num'] = df_quartier_clean['Zimmeranzahl'].apply(zimmer_zu_int)
df_baualter_clean['Zimmeranzahl_num'] = df_baualter_clean['Zimmeranzahl'].apply(zimmer_zu_int)

# Baualter in numerisches Format umwandeln
def baualter_zu_jahr(baualter_str):
    """Wandelt Baualter-Text in ungefähres Baujahr um"""
    try:
        # Format "1981-2000"
        if '-' in baualter_str:
            jahre = baualter_str.split('-')
            return (int(jahre[0]) + int(jahre[1])) / 2
        # Format "vor 1919"
        elif 'vor' in baualter_str:
            return 1919
        # Format "nach 2015" oder "seit 2015"
        elif 'nach' in baualter_str or 'seit' in baualter_str:
            return 2015
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
    'MedianPreis': 'mean',
    'Baujahr': 'mean'
}).reset_index()

df_baualter_agg.rename(columns={
    'MedianPreis': 'MedianPreis_Baualter',
    'Baujahr': 'Durchschnitt_Baujahr'
}, inplace=True)

# Die aggregierten Baualter-Daten mit dem Quartier-Datensatz verbinden
df_merged = pd.merge(
    df_quartier_clean,
    df_baualter_agg,
    on=['Jahr', 'Zimmeranzahl_num'],
    how='left'
)

# Einige zusätzliche Features erstellen
# Preisverhältnis: Quartierpreis zu durchschnittlichem Preis nach Baualter
df_merged['Preis_Verhältnis'] = df_merged['MedianPreis'] / df_merged['MedianPreis_Baualter']

# Neustes Jahr für das finale Modelltraining wählen
neuestes_jahr = df_merged['Jahr'].max()
df_final = df_merged[df_merged['Jahr'] == neuestes_jahr].copy()

# Feature für Quartier-Preisniveau: Durchschnittlicher Preis im Quartier relativ zum Gesamtdurchschnitt
quartier_avg_preis = df_final.groupby('Quartier')['MedianPreis'].mean()
gesamtpreis_avg = quartier_avg_preis.mean()
quartier_preisniveau = (quartier_avg_preis / gesamtpreis_avg).to_dict()

df_final['Quartier_Preisniveau'] = df_final['Quartier'].map(quartier_preisniveau)

# Bereinigen: Zeilen mit fehlenden Werten entfernen oder ersetzen
df_final.dropna(subset=['MedianPreis', 'Quartier', 'Zimmeranzahl_num'], inplace=True)

# Restliche NaN-Werte durch sinnvolle Werte ersetzen
for column in df_final.columns:
    if df_final[column].dtype in [np.float64, np.int64]:
        df_final[column].fillna(df_final[column].median(), inplace=True)

# Kategorische Features vorbereiten (für ML-Modelle wie Random Forest)
# Quartier als kategorisches Feature - später One-Hot-Encoding anwenden
df_final['Quartier_Code'] = pd.Categorical(df_final['Quartier']).codes

# Daten speichern
df_quartier_clean.to_csv('data/processed/quartier_processed.csv', index=False)
df_baualter_clean.to_csv('data/processed/baualter_processed.csv', index=False)
df_final.to_csv('data/processed/modell_input_final.csv', index=False)