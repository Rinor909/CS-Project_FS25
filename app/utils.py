# Notwendige Bibliotheken importieren
import pandas as pd         # Für Datenverarbeitung und -analyse
import numpy as np          # Für numerische Berechnungen
import pickle              # Für Speicherung/Ladung von Modellen
import os                  # Für Dateisystemoperationen
import requests            # Für HTTP-Anfragen
from io import StringIO    # Für String-basierte Datenströme

def load_processed_data():
   """Lädt die verarbeiteten Daten für die Anwendung"""
   # GitHub-URLs für Datenquellen
   quartier_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/quartier_processed.csv"
   baualter_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/baualter_processed.csv"
   travel_times_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/travel_times.csv"
   
   # Quartierdaten laden
   try:
       # Versuch, Daten direkt von GitHub zu laden
       df_quartier = pd.read_csv(quartier_url)
   except:
       # Fallback auf lokale Datei oder Erstellung eines leeren DataFrames
       if os.path.exists('data/processed/quartier_processed.csv'):
           df_quartier = pd.read_csv('data/processed/quartier_processed.csv')
       else:
           # Erstellt leeres DataFrame mit erwarteten Spalten
           df_quartier = pd.DataFrame(columns=['Jahr', 'Quartier', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num'])
           df_quartier['Jahr'] = [2024]  # Fügt Standardjahr hinzu
   
   # Baualter-Daten laden
   try:
       # Versuch, Daten direkt von GitHub zu laden
       df_baualter = pd.read_csv(baualter_url)
   except:
       # Fallback auf lokale Datei oder Erstellung eines leeren DataFrames
       if os.path.exists('data/processed/baualter_processed.csv'):
           df_baualter = pd.read_csv('data/processed/baualter_processed.csv')
       else:
           # Erstellt leeres DataFrame mit erwarteten Spalten
           df_baualter = pd.DataFrame(columns=['Jahr', 'Baualter', 'Zimmeranzahl', 'MedianPreis', 'PreisProQm', 'Zimmeranzahl_num', 'Baujahr'])
   
   # Reisezeit-Daten laden
   try:
       # Versuch, Daten direkt von GitHub zu laden
       df_travel_times = pd.read_csv(travel_times_url)
   except:
       # Fallback auf lokale Datei oder Erstellung eines leeren DataFrames
       if os.path.exists('data/processed/travel_times.csv'):
           df_travel_times = pd.read_csv('data/processed/travel_times.csv')
       else:
           # Erstellt leeres DataFrame mit erwarteten Spalten
           df_travel_times = pd.DataFrame(columns=['Quartier', 'Zielort', 'Transportmittel', 'Reisezeit_Minuten'])
   
   # Stellt sicher, dass die Jahr-Spalte existiert
   if 'Jahr' not in df_quartier.columns:
       df_quartier['Jahr'] = 2024
       
   return df_quartier, df_baualter, df_travel_times

def load_model():
   """Lädt das trainierte Preisvorhersagemodell"""
   # URL zum Modell auf GitHub
   model_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/models/price_model.pkl"
   
   try:
       # Versuch, Modell von GitHub zu laden
       response = requests.get(model_url)
       if response.status_code == 200:
           # Konvertiert die Antwort in ein Modellobjekt
           model_data = StringIO(response.content.decode('latin1'))
           model = pickle.load(model_data)
           return model
       
       # Fallback auf lokale Datei
       if os.path.exists('models/price_model.pkl'):
           with open('models/price_model.pkl', 'rb') as file:
               model = pickle.load(file)
           return model
   except:
       # Bei Fehler stillschweigend fortfahren
       pass
   
   # Wenn das Modell nicht geladen werden kann, gibt None zurück (Fallback-Berechnung wird verwendet)
   return None

def load_quartier_mapping():
   """Lädt die Quartier-Mapping-Daten"""
   try:
       # Versucht, das Mapping aus lokaler Datei zu laden
       if os.path.exists('models/quartier_mapping.pkl'):
           with open('models/quartier_mapping.pkl', 'rb') as file:
               return pickle.load(file)
   except:
       # Bei Fehler stillschweigend fortfahren
       pass
   
   # Leeres Mapping als Standardwert, wenn Laden fehlschlägt
   return {}

def preprocess_input(quartier_code, zimmeranzahl, baujahr, travel_times_dict):
   """Bereitet Eingabedaten für das Modell vor"""
   # Modell-Eingabedaten laden
   try:
       # Versuch, Daten von GitHub zu laden
       model_input_url = "https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv"
       df_final = pd.read_csv(model_input_url)
   except:
       # Versuche lokale Datei
       try:
           if os.path.exists('data/processed/modell_input_final.csv'):
               df_final = pd.read_csv('data/processed/modell_input_final.csv')
           else:
               # Wirft Ausnahme, wenn Datei nicht existiert
               raise Exception()
       except:
           # Standardwerte, wenn Laden fehlschlägt
           df_final = pd.DataFrame({
               'Quartier_Code': [0],
               'Quartier_Preisniveau': [1.0],
               'MedianPreis_Baualter': [1000000]
           })
   
   # Werte für das ausgewählte Quartier finden
   quartier_data = df_final[df_final['Quartier_Code'] == quartier_code] if 'Quartier_Code' in df_final.columns else pd.DataFrame()
   
   # Ermittelt Preisniveau und Median-Baualterspreis für das ausgewählte Quartier
   if len(quartier_data) > 0:
       # Wenn Quartierdaten verfügbar sind, verwende diese
       quartier_preisniveau = quartier_data['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in quartier_data.columns else 1.0
       mediapreis_baualter = quartier_data['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in quartier_data.columns else 1000000
   else:
       # Fallback: Durchschnittswerte oder Standardwerte verwenden
       quartier_preisniveau = df_final['Quartier_Preisniveau'].mean() if 'Quartier_Preisniveau' in df_final.columns else 1.0
       mediapreis_baualter = df_final['MedianPreis_Baualter'].mean() if 'MedianPreis_Baualter' in df_final.columns else 1000000
   
   # Eingabedaten erstellen
   input_data = pd.DataFrame({
       'Quartier_Code': [quartier_code],
       'Zimmeranzahl_num': [zimmeranzahl],
       'PreisProQm': [quartier_preisniveau * 10000],  # Approximation
       'MedianPreis_Baualter': [mediapreis_baualter],
       'Durchschnitt_Baujahr': [baujahr],
       'Preis_Verhältnis': [1.0],  # Standardwert
       'Quartier_Preisniveau': [quartier_preisniveau]
   })
   
   # Reisezeiten hinzufügen, falls verfügbar
   for key, value in travel_times_dict.items():
       if key in ['Hauptbahnhof', 'ETH', 'Flughafen', 'Bahnhofstrasse']:
           input_data[f'Reisezeit_{key}'] = value
   
   return input_data

def predict_price(model, input_data):
   """Sagt den Preis basierend auf Eingabedaten voraus"""
   # Schlüsselwerte für Fallback-Berechnungen extrahieren
   quartier_code = input_data['Quartier_Code'].values[0] if 'Quartier_Code' in input_data.columns else 0
   zimmeranzahl = input_data['Zimmeranzahl_num'].values[0] if 'Zimmeranzahl_num' in input_data.columns else 3
   baujahr = input_data['Durchschnitt_Baujahr'].values[0] if 'Durchschnitt_Baujahr' in input_data.columns else 2000
   quartier_preisniveau = input_data['Quartier_Preisniveau'].values[0] if 'Quartier_Preisniveau' in input_data.columns else 1.0
   
   # Fallback-Berechnungsfunktion
   def calculate_fallback_price():
       # Basispreis basierend auf Quartier-Preisniveau
       base_price = 1000000 * quartier_preisniveau
       
       # Zimmerfaktor: Jedes Zimmer erhöht den Basispreis um 15%
       room_factor = 1.0 + ((zimmeranzahl - 3) * 0.15)
       
       # Altersfaktor: Neuere Gebäude sind teurer
       aktuelles_jahr = 2025
       max_age = 100
       age = max(0, min(max_age, aktuelles_jahr - baujahr))
       age_factor = 1.3 - (age / max_age * 0.6)  # Bereich von 0,7 bis 1,3
       
       # Berechnet Gesamtpreis durch Multiplikation der Faktoren
       return base_price * room_factor * age_factor
   
   # Wenn kein Modell vorhanden, verwende Fallback-Berechnung
   if model is None:
       return calculate_fallback_price()
   
   try:
       # Vorhersage mit Modell treffen
       prediction = model.predict(input_data)[0]
       
       # Plausibilitätsprüfung - wenn Vorhersage unvernünftig erscheint, basierend auf Baujahr anpassen
       if prediction < 500000 or prediction > 10000000:
           aktuelles_jahr = 2025
           max_age = 100
           age = max(0, min(max_age, aktuelles_jahr - baujahr))
           age_factor = 1.3 - (age / max_age * 0.6)
           prediction = prediction * age_factor
       
       return round(prediction, 2)
   except:
       # Wenn Vorhersage fehlschlägt, verwende Fallback-Berechnung
       return calculate_fallback_price()

def get_travel_times_for_quartier(quartier, df_travel_times, transportmittel='transit'):
   """Gibt Reisezeiten für ein bestimmtes Quartier zurück"""
   # Standard-Reisezeiten als Fallback
   default_times = {
       'Hauptbahnhof': 20,
       'ETH': 25,
       'Flughafen': 35,
       'Bahnhofstrasse': 22
   }
   
   # Standardwerte zurückgeben, wenn Daten fehlen oder leer sind
   if df_travel_times.empty or 'Quartier' not in df_travel_times.columns:
       return default_times
   
   # Daten für das ausgewählte Quartier und Transportmittel filtern
   filtered_data = df_travel_times[
       (df_travel_times['Quartier'] == quartier) & 
       (df_travel_times['Transportmittel'] == transportmittel)
   ]
   
   # Reisezeiten-Dictionary erstellen
   travel_times = {}
   for _, row in filtered_data.iterrows():
       travel_times[row['Zielort']] = row['Reisezeit_Minuten']
   
   # Standardwerte zurückgeben, wenn keine passenden Daten gefunden wurden
   return travel_times if travel_times else default_times

def get_quartier_statistics(quartier, df_quartier):
   """Berechnet Statistiken für ein bestimmtes Quartier"""
   # Standardstatistiken
   default_stats = {
       'median_preis': 1000000,
       'min_preis': 800000,
       'max_preis': 1200000,
       'preis_pro_qm': 10000,
       'anzahl_objekte': 0
   }
   
   # Standardwerte zurückgeben, wenn Daten fehlen
   if 'Quartier' not in df_quartier.columns or df_quartier.empty:
       return default_stats
   
   # Daten für ausgewähltes Quartier filtern
   quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
   
   # Standardwerte zurückgeben, wenn keine passenden Daten gefunden wurden
   if quartier_data.empty:
       return default_stats
   
   # Statistiken berechnen
   return {
       'median_preis': quartier_data['MedianPreis'].median() if 'MedianPreis' in quartier_data.columns else default_stats['median_preis'],
       'min_preis': quartier_data['MedianPreis'].min() if 'MedianPreis' in quartier_data.columns else default_stats['min_preis'],
       'max_preis': quartier_data['MedianPreis'].max() if 'MedianPreis' in quartier_data.columns else default_stats['max_preis'],
       'preis_pro_qm': quartier_data['PreisProQm'].median() if 'PreisProQm' in quartier_data.columns else default_stats['preis_pro_qm'],
       'anzahl_objekte': len(quartier_data)
   }

def get_price_history(quartier, df_quartier):
   """Gibt die Preisentwicklung für ein bestimmtes Quartier zurück"""
   # Leeres DataFrame zurückgeben, wenn Daten fehlen
   if 'Quartier' not in df_quartier.columns or 'Jahr' not in df_quartier.columns or df_quartier.empty:
       return pd.DataFrame()
   
   # Daten für ausgewähltes Quartier filtern
   quartier_data = df_quartier[df_quartier['Quartier'] == quartier]
   
   # Leeres DataFrame zurückgeben, wenn keine passenden Daten gefunden wurden
   if quartier_data.empty:
       return pd.DataFrame()
   
   # Nach Jahr gruppieren und Medianpreise berechnen
   price_history = quartier_data.groupby('Jahr').agg({
       'MedianPreis': 'median',
       'PreisProQm': 'median' if 'PreisProQm' in quartier_data.columns else lambda x: None
   }).reset_index()
   
   return price_history

def get_zurich_coordinates():
   """Gibt Koordinaten für Zürich zurück"""
   return {
       'latitude': 47.3769,
       'longitude': 8.5417,
       'zoom': 12
   }

def get_quartier_coordinates():
   """Gibt Koordinaten für alle Quartiere in Zürich zurück"""
   return {
       'Hottingen': {'lat': 47.3692, 'lng': 8.5631},
       'Fluntern': {'lat': 47.3809, 'lng': 8.5629},
       'Unterstrass': {'lat': 47.3864, 'lng': 8.5419},
       'Oberstrass': {'lat': 47.3889, 'lng': 8.5481},
       'Rathaus': {'lat': 47.3716, 'lng': 8.5428},
       'Lindenhof': {'lat': 47.3728, 'lng': 8.5408},
       'City': {'lat': 47.3752, 'lng': 8.5385},
       'Seefeld': {'lat': 47.3600, 'lng': 8.5532},
       'Mühlebach': {'lat': 47.3638, 'lng': 8.5471},
       'Witikon': {'lat': 47.3610, 'lng': 8.5881},
       'Hirslanden': {'lat': 47.3624, 'lng': 8.5705},
       'Enge': {'lat': 47.3628, 'lng': 8.5288},
       'Wollishofen': {'lat': 47.3489, 'lng': 8.5266},
       'Leimbach': {'lat': 47.3279, 'lng': 8.5098},
       'Friesenberg': {'lat': 47.3488, 'lng': 8.5035},
       'Alt-Wiedikon': {'lat': 47.3652, 'lng': 8.5158},
       'Sihlfeld': {'lat': 47.3742, 'lng': 8.5072},
       'Albisrieden': {'lat': 47.3776, 'lng': 8.4842},
       'Altstetten': {'lat': 47.3917, 'lng': 8.4876},
       'Höngg': {'lat': 47.4023, 'lng': 8.4976},
       'Wipkingen': {'lat': 47.3930, 'lng': 8.5253},
       'Affoltern': {'lat': 47.4230, 'lng': 8.5047},
       'Oerlikon': {'lat': 47.4126, 'lng': 8.5487},
       'Seebach': {'lat': 47.4258, 'lng': 8.5422},
       'Saatlen': {'lat': 47.4087, 'lng': 8.5742},
       'Schwamendingen-Mitte': {'lat': 47.4064, 'lng': 8.5648},
       'Hirzenbach': {'lat': 47.4031, 'lng': 8.5841}
   }