import pandas as pd
import numpy as np
import os

# Dieses Skript bereitet die Daten für die Modellierung vor, indem es sie bereinigt, transformiert und speichert.
# KI-Tools wurden verwendet, um das Skript zu strukturieren, es zu debuggen und einige anspruchsvollere Teile 
# des Codes zu schreiben (die unten erklärt werden), um die Genauigkeit und Effizienz zu verbessern.

# Hier gebe ich ein lokales Ausgabeverzeichnis an, da ich nicht in der Lage war, 
# sie direkt auf GitHub zu speichern (ich habe 2 Tage lang versucht, dies zu beheben, ohne Erfolg).
# Ich speichere also die Exporte lokal und lade sie dann auf GitHub hoch, um sie später für die anderen Skripte zu verwenden.
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data" # Ich lege fest, wo die Datei gespeichert werden soll
processed_dir = os.path.join(output_dir, "processed") # Erzeugt einen Unterordner mit dem Namen „processed“

# Ich stelle sicher, dass die Datenverzeichnisse existieren oder erstelle sie falls nicht
# processed_dir: zum Speichern bereinigter/transformierter CSV-Datendateien
# models-Verzeichnis: zum Speichern trainierter Machine-Learning-Modelldateien
os.makedirs(processed_dir, exist_ok=True) # Legt diese Ordner an, wenn sie nicht existieren (exist_ok=True verhindert Fehler, wenn die Ordner bereits existieren)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

# Die CSV-Dateien werden direkt von den Raw-URLs unseres GitHub-Repositories geladen,
# da der lokale Dateizugriff auf verschiedenen Systemen Probleme verursachte
url_quartier = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/raw/bau515od5155.csv' # Definiert die erste URL, die auf die erste CSV-Datei auf GitHub verweist
url_baualter = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/raw/bau515od5156.csv' # Definiert die zweite URL, die auf die zweite CSV-Datei auf GitHub verweist
df_quartier = pd.read_csv(url_quartier) # Wir verwenden pandas, um diese CSV-Dateien direkt in ein DataFrame zu laden
df_baualter = pd.read_csv(url_baualter) # Wir verwenden pandas, um diese CSV-Dateien direkt in ein DataFrame zu laden

# Wichtige Spalten aus dem Quartier-Datensatz auswählen und umbenennen
# Diese Extraktion verbessert die Lesbarkeit und vereinfacht die Weiterverarbeitung
quartier_spalten = { # Wir erstellen ein Wörterbuch, um die ursprünglichen Spaltennamen in neue, besser verständliche Namen zuzuordnen
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'RaumLang': 'Quartier',                 # Name des Stadtquartiers 
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_quartier_clean = df_quartier[quartier_spalten.keys()].copy() # Wir erstellen ein neues DataFrame, das nur die ausgewählten Spalten enthält
df_quartier_clean.rename(columns=quartier_spalten, inplace=True) # Wir benennen diese Spalten in deutsche Namen um

# Ähnliche Transformation für den Baualter-Datensatz durchführen
baualter_spalten = { # Erstelle ein Wörterbuch zur Zuordnung der ursprünglichen Spalten zu neuen Namen
    'Stichtagdatjahr': 'Jahr',              # Jahr der Datenerhebung
    'BaualterLang_noDM': 'Baualter',        # Baualtersklasse als Text
    'AnzZimmerLevel2Lang_noDM': 'Zimmeranzahl',  # Zimmeranzahl als Text
    'HAMedianPreis': 'MedianPreis',         # Median-Verkaufspreis
    'HAPreisWohnflaeche': 'PreisProQm'      # Preis pro Quadratmeter
}

df_baualter_clean = df_baualter[baualter_spalten.keys()].copy() # Wir erstellen ein neues DataFrame, das nur die ausgewählten Spalten enthält
df_baualter_clean.rename(columns=baualter_spalten, inplace=True) # Wir benennen diese Spalten in deutsche Namen um
# Fehlende Werte durch kontextabhängige Mediane oder Mittelwerte ersetzen
# Dies erhöht die Datenqualität und verhindert Verzerrungen durch fehlende Werte
# Für den Code zwischen Zeile 52 und 75 wurden KI-Tools verwendet, um eine robuste Fehlerbehandlung zu gewährleisten, da die Datensätze viele fehlende Werte und nur wenige Datenpunkte enthielten
for df in [df_quartier_clean, df_baualter_clean]: # Wir verarbeiten beide Datensätze in einer Schleife
    if 'Quartier' in df.columns: # Beim ersten Datensatz werden fehlende Preiswerte mit dem am besten geeigneten Ersatzwert aufgefüllt
        # Für Quartier-Datensatz: Gruppierung nach Quartier und Zimmeranzahl
        # MedianPreis: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Quartier', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
        
        # PreisProQm: Fehlende Werte nach Quartier und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Quartier', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
    else: # Für den zweiten Datensatz werden fehlende Preiswerte mit dem am besten geeigneten Ersatzwert aufgefüllt
        # Für Baualter-Datensatz: Gruppierung nach Baualter und Zimmeranzahl
        # MedianPreis: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['MedianPreis'] = df.groupby(['Baualter', 'Zimmeranzahl'])['MedianPreis'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
        
        # PreisProQm: Fehlende Werte nach Baualter und Zimmeranzahl ersetzen
        df['PreisProQm'] = df.groupby(['Baualter', 'Zimmeranzahl'])['PreisProQm'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean() if not pd.isna(x.mean()) else 0))
    
    # Globale Mediane für verbleibende fehlende Werte verwenden
    median_price = df['MedianPreis'].median() # Wir berechnen den Gesamtmedian der Preise im gesamten Datensatz

    df['MedianPreis'].fillna(0 if pd.isna(median_price) else median_price, inplace=True) # Wir berechnen den Gesamtmedian der Preise im gesamten Datensatz
    median_price_per_sqm = df['PreisProQm'].median() # Wir berechnen den Gesamtmedian des Quadratmeterpreises im gesamten Datensatz
    df['PreisProQm'].fillna(0 if pd.isna(median_price_per_sqm) else median_price_per_sqm, inplace=True) # Wie zuvor: Median verwenden, um Lücken zu füllen, andernfalls 0 einsetzen

# Feature-Engineering: Textbasierte Zimmeranzahl in numerischen Wert konvertieren
# Dies ermöglicht mathematische Operationen und ML-Modellierung
def zimmer_zu_int(zimmer_str):
    """Extrahiert die Zimmeranzahl aus dem String-Format"""
    try:
        return int(zimmer_str.split('-')[0]) # Extrahiert '3' aus '3-Zimmer'
    except:
        return np.nan # andernfalls leeren Wert zurückgeben

df_quartier_clean['Zimmeranzahl_num'] = df_quartier_clean['Zimmeranzahl'].apply(zimmer_zu_int) # Wir verwenden die Methode .apply(), um diese Funktion auf jeden Wert der Spalte Zimmeranzahl in unserem bereinigten Datensatz anzuwenden
df_baualter_clean['Zimmeranzahl_num'] = df_baualter_clean['Zimmeranzahl'].apply(zimmer_zu_int) # Wir verwenden die Methode .apply(), um diese Funktion auf jeden Wert der Spalte Zimmeranzahl in unserem bereinigten Datensatz anzuwenden
# Feature-Engineering: Baualter-Kategorien in geschätztes Baujahr umwandeln
# Wandelt kategorische Variable in kontinuierlichen Wert für ML-Modellierung um
# KI-Tools wurden verwendet, um diese Funktion zu erstellen, da sie eine komplexe Logik zur Verarbeitung von Textwerten enthält
def baualter_zu_jahr(baualter_str):
    """Wandelt Baualter-Text in ungefähres Baujahr um"""
    try:
        # Zeitspannen in einen Mittelwert umwandeln (z.B. '1981-2000' → 1990.5)
        if '-' in baualter_str: # Wenn ein - vorhanden ist: Teile diesen Wert nach dem - in zwei Hälften und nimm den Durchschnitt davon.
            jahre = baualter_str.split('-')
            return (int(jahre[0]) + int(jahre[1])) / 2
        # Spezielle Kategorien interpretieren
        elif 'vor' in baualter_str: # falls ein Wert „vor 1919“ angegeben ist, gib 1919 zurück
            return 1919 # Standardwert für 'vor 1919'
        # Format "nach 2015" oder "seit 2015"
        elif 'nach' in baualter_str or 'seit' in baualter_str: # Bei neueren Gebäuden mit den Angaben nach oder seit im Wert wird 2015 zurückgegeben
            return 2015 # Standardwert für neuere Gebäude
        else:
            return np.nan # andernfalls leeren Wert zurückgeben
    except:
        return np.nan # andernfalls leeren Wert zurückgeben

if 'Baualter' in df_baualter_clean.columns:
    df_baualter_clean['Baujahr'] = df_baualter_clean['Baualter'].apply(baualter_zu_jahr) # Erstellen einer neuen Spalte 'Baujahr' mit diesen geschätzten Werten.

# Gebäudealter-Features zum Quartier-Datensatz hinzufügen
# Für jedes Quartier, Jahr und Zimmeranzahl den Durchschnittspreis pro Baualtersklasse ermitteln
# und dann mit dem Quartier-Datensatz zusammenführen

# Aggregieren der Baualter-Daten nach Jahr und Zimmeranzahl
df_baualter_agg = df_baualter_clean.groupby(['Jahr', 'Zimmeranzahl_num']).agg({ # Gruppiert die Daten zum Gebäudealter nach Jahr und Anzahl der Zimmer.
    'MedianPreis': 'mean', # Durchschnittspreis pro Baualter-Kategorie
    'Baujahr': 'mean'   # Durchschnittliches Baujahr
}).reset_index() # Setzt den Index zurück, um das gruppierte Ergebnis wieder in einen regulären DataFrame umzuwandeln.

# Spalten für bessere Lesbarkeit umbenennen
df_baualter_agg.rename(columns={ # Wir benennen die Spalten um, um anzuzeigen, dass es sich um aggregierte Werte handelt
    'MedianPreis': 'MedianPreis_Baualter',
    'Baujahr': 'Durchschnitt_Baujahr'
}, inplace=True)

# Quartier- und Baualter-Daten mittels Jahr und Zimmeranzahl zusammenführen
# Dadurch werden beide Dimensionen in einem gemeinsamen Datensatz vereint
df_merged = pd.merge( # Wir verbinden die Quartiersdaten mit den aggregierten Baualtersdaten
    df_quartier_clean,
    df_baualter_agg,
    on=['Jahr', 'Zimmeranzahl_num'], # Wir ordnen die Zeilen anhand von Jahr und Zimmeranzahl_num zu
    how='left' # Wir verwenden einen Left Join, der alle Zeilen des Quartiersdatensatzes beibehält und, falls vorhanden, Baualtersdaten hinzufügt
)

# Feature-Engineering: Preisverhältnis zwischen Quartier und Baualter berechnen
# Dieses Feature zeigt die relative Preispositionierung eines Quartiers im Vergleich zum Baujahrstandard
df_merged['Preis_Verhältnis'] = df_merged['MedianPreis'] / df_merged['MedianPreis_Baualter'] # Wir teilen den Medianpreis jedes Quartiers durch den durchschnittlichen Medianpreis von Objekten ähnlichen Alters und ähnlicher Zimmeranzahl
# Nur die neuesten Daten für das Modelltraining filtern
# Aktuelle Daten sind relevanter für Preisvorhersagen
neuestes_jahr = df_merged['Jahr'].max() # Findet das aktuellste Jahr im Datensatz.
df_final = df_merged[df_merged['Jahr'] == neuestes_jahr].copy() # Erstellen eines neuen DataFrames, das nur die Daten des aktuellsten Jahres enthält
# Feature-Engineering: Relatives Preisniveau pro Quartier berechnen
# Ermöglicht Vergleich zwischen Quartieren unabhängig von absoluten Preisen
quartier_avg_preis = df_final.groupby('Quartier')['MedianPreis'].mean() # Wir berechnen den Durchschnittspreis für jedes Quartier
gesamtpreis_avg = quartier_avg_preis.mean() # Wir berechnen den Durchschnittspreis für jedes Quartier
quartier_preisniveau = (quartier_avg_preis / gesamtpreis_avg).to_dict() # Wir berechnen den Durchschnittspreis für jedes Quartier

df_final['Quartier_Preisniveau'] = df_final['Quartier'].map(quartier_preisniveau) # Wir wandeln dies in ein Wörterbuch um, das Quartiersnamen den Preisniveaus zuordnet, und fügen diesen relativen Preiswert wieder in jede Zeile des Datensatzes ein
# Zeilen mit fehlenden Werten in Kernvariablen entfernen
# Sichert die Datenqualität für das ML-Modell
df_final.dropna(subset=['MedianPreis', 'Quartier', 'Zimmeranzahl_num'], inplace=True) # Wir berechnen den Gesamtdurchschnittspreis aller Quartiere

# Verbleibende fehlende Werte durch Mediane ersetzen
# Sicherstellen, dass das Dataset vollständig ist, ohne Informationsverlust
for column in df_final.columns:
    if df_final[column].dtype in [np.float64, np.int64]:
        # Robuste Behandlung von Spalten, die nur NaN-Werte enthalten könnten
        df_final = df_final.copy() 
        median_value = df_final[column].median() 
        if pd.isna(median_value):
            # Fallback auf 0 wenn kein Median berechnet werden kann
            df_final.loc[:, column] = df_final[column].fillna(0) # andernfalls setzen wir den Wert auf 0, wenn kein Median berechnet werden konnte
        else:
            # Standardfall: Mit Median füllen
            df_final.loc[:, column] = df_final[column].fillna(median_value) # Wir füllen alle verbleibenden fehlenden Werte in numerischen Spalten mit dem Medianwert der jeweiligen Spalte auf

# Quartiere für ML-Modelle in numerische Codes konvertieren
# One-Hot-Encoding-Vorbereitung für kategorische Variablen
df_final['Quartier_Code'] = pd.Categorical(df_final['Quartier']).codes # Wir erstellen eine kategoriale Kodierung der Quartiernamen, eine neue Spalte mit numerischen Codes, die jedes Quartier repräsentieren, wodurch die Daten für Machine-Learning-Algorithmen vorbereitet werden, die numerische Eingaben erfordern.

# Aufbereitete Datensätze in CSV-Dateien speichern
# Diese Dateien werden für Modelltraining und App-Visualisierung verwendet
quartier_path = os.path.join(processed_dir, 'quartier_processed.csv') # Wir definieren Dateipfade zum Speichern der verarbeiteten Datensätze.
baualter_path = os.path.join(processed_dir, 'baualter_processed.csv')
final_path = os.path.join(processed_dir, 'modell_input_final.csv')

df_quartier_clean.to_csv(quartier_path, index=False) # Wir speichern alle drei Datensätze als CSV-Dateien
df_baualter_clean.to_csv(baualter_path, index=False) # Wir verwenden index=False, um zu verhindern, dass der DataFrame-Index als zusätzliche Spalte gespeichert wird
df_final.to_csv(final_path, index=False) # Wir verwenden index=False, um zu verhindern, dass der DataFrame-Index als zusätzliche Spalte gespeichert wird
