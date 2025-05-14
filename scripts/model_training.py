import pandas as pd # Datenmanipulationsbibliothek
import numpy as np # Datenmanipulationsbibliothek
import pickle # Zum Speichern trainierter Modelle in Dateien
import os # Dateisystemoperationen
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Machine-Learning-Modelle und Hilfsfunktionen
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Wie zuvor in data_preparation.py und generate_travel_times.py definieren wir erneut das Ausgabeverzeichnis
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed") # Zum Speichern von Zwischendateien
models_dir = os.path.join(output_dir, "models") # Zum Speichern trainierter ML-Modelle und zugehöriger Dateien
os.makedirs(processed_dir, exist_ok=True) # Verzeichnisse erstellen, falls sie nicht existieren
os.makedirs(models_dir, exist_ok=True) # Verzeichnisse erstellen, falls sie nicht existieren

# Wir verwenden erneut die direkte GitHub-URL, um die CSV-Datei zu laden
url_modell_input_final = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv'
try: # Versuch, die Daten zuerst aus dem Git-Repository zu laden
    df = pd.read_csv('https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv')
except: # Rückgriff auf lokale Datei, falls Git-Repository fehlschlägt
    df = pd.read_csv(os.path.join(processed_dir, 'modell_input_final.csv'))

# Feature und Zielwerte definieren
# Features: Quartier, Zimmeranzahl, Preisniveau, Baujahr, etc.
# Ziel: MedianPreis
X = df.drop(['MedianPreis', 'Quartier', 'Zimmeranzahl'], axis=1) # Enthält alle Merkmale für die Vorhersage; entfernt Zielvariable und redundante Spalten
y = df['MedianPreis'] # Zielvariable

# Kategoriale und numerische Features identifizieren # necessary separation for proper processing in sklearn pipelines
cat_features = ['Quartier_Code'] # Notwendige Trennung für korrekte Verarbeitung in sklearn-Pipelines
num_features = [col for col in X.columns if col not in cat_features] # Alle anderen Variablen sind numerisch

# Train-Test-Split (80/20) # Trainingsdaten werden zum Erstellen des Modells verwendet; Testdaten dienen zur Bewertung der Modellleistung
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80 % Trainings- und 20 % Testdaten; random_state = 42 sorgt für reproduzierbare Ergebnisse

# Modell-Pipeline erstellen mit One-Hot-Encoding für kategoriale Features # handles different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features), # Numerische Merkmale bleiben unverändert
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features) # One-Hot-Codierung (Umwandlung in binäre Spalten); handle_unknown behandelt unbekannte Kategorien bei der Vorhersage
    ])

# Da im Kurs Random Forests und Gradient Boosting nur unzureichend erklärt wurden, habe ich diesen Teil mithilfe von KI programmiert
# Modelle definieren
# Random Forest Regressor; ein Ensemble aus Entscheidungsbäumen, die auf zufälligen Teildatensätzen trainiert wurden
rf_model = Pipeline([
    ('preprocessor', preprocessor), # Jede Pipeline kombiniert Vorverarbeitung und Regressor
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)) # n_estimators bestimmt die Anzahl der Bäume im Ensemble; random_state gewährleistet Reproduzierbarkeit
])

# Gradient Boosting Regressor # Gradient Boosting Regressor; ein Ensembleverfahren, das Bäume sequentiell erstellt
gb_model = Pipeline([
    ('preprocessor', preprocessor), # Jede Pipeline kombiniert Vorverarbeitung und Regressor
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42)) # n_estimators bestimmt die Anzahl der Bäume im Ensemble; random_state gewährleistet Reproduzierbarkeit
])

# Modelle trainieren # models learn patterns that relate features to property prices # most intensive part of the code
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Funktion zur Bewertung des Modells
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred) # Durchschnittlicher absoluter Unterschied zwischen Vorhersagen und tatsächlichen Werten
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Quadratwurzel des durchschnittlichen quadrierten Fehlers
    r2 = r2_score(y_test, y_pred) # Anteil der erklärten Varianz durch das Modell
    return mae, rmse, r2

# Bewertung der Modelle durch Anwendung der Funktion auf beide; Speichern der Leistungskennzahlen zum Vergleich
rf_metrics = evaluate_model(rf_model, X_test, y_test) 
gb_metrics = evaluate_model(gb_model, X_test, y_test)

# Testet das Modell auf mehreren Trainings-/Testaufteilungen, um Robustheit sicherzustellen
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error') # 5-fache Kreuzvalidierung für beide Modelle; mittlerer quadratischer Fehler als Bewertungsmaß
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error') # Gibt negative Werte zurück, die später in positive umgewandelt werden

# Feature Importance für Random Forest Modell
if hasattr(rf_model.named_steps['regressor'], 'feature_importances_'): # Extrahiert die Feature-Importance aus dem Random Forest Modell
    # Column names nach der Transformation durch OneHotEncoder
    ohe = preprocessor.named_transformers_['cat'] 
    cat_cols = ohe.get_feature_names_out(['Quartier_Code'])
    feature_names = num_features + list(cat_cols) # Holt sich die umgewandelten Spaltennamen nach One-Hot-Encoding 
    # Feature Importance extrahieren
    importances = rf_model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1] # Sortiert die Merkmale nach Wichtigkeit (absteigend), um die einflussreichsten Faktoren auf Immobilienpreise zu erkennen

# Beste Modell auswählen
best_model = rf_model if rf_metrics[2] > gb_metrics[2] else gb_model # Vergleicht Modelle anhand der R^2-Leistung (rf_metrics[2])
best_model_name = "Random Forest" if rf_metrics[2] > gb_metrics[2] else "Gradient Boosting" # Wählt das Modell mit der höchsten R^2-Leistung als bestes Modell; speichert den Namen für Dokumentationszwecke

# Modell speichern 
with open(os.path.join(models_dir, 'price_model.pkl'), 'wb') as file: # wb öffnet die Datei im binären Schreibmodus
    pickle.dump(best_model, file) # Speichert das beste Modell in einer Pickle-Datei, um es später ohne erneutes Training wiederzuverwenden

# Quartier-Mapping für die Anwendung speichern
quartier_mapping = {code: quartier for code, quartier in zip(df['Quartier_Code'], df['Quartier'])} # Erstellt ein Wörterbuch, das numerische Codes den Quartiersnamen zuordnet
with open(os.path.join(models_dir, 'quartier_mapping.pkl'), 'wb') as file: 
    pickle.dump(quartier_mapping, file) # save this mapping to a pickle file # Speichert dieses Mapping in einer Pickle-Datei; hilft, die Modellausgabe in lesbare Quartiersnamen zurückzuübersetzen

# Save model metrics
with open(os.path.join(models_dir, 'model_metrics.txt'), 'w') as file: # Speichert Leistungskennzahlen in einer .txt-Datei
    file.write(f"Model: {best_model_name}\n") # Gibt an, welches Modell ausgewählt wurde und zeigt anschließend dessen Leistungskennzahlen an; formatiert Werte mit entsprechender Genauigkeit
    file.write(f"MAE: {rf_metrics[0]:.2f} CHF\n" if best_model_name == "Random Forest" else f"MAE: {gb_metrics[0]:.2f} CHF\n")
    file.write(f"RMSE: {rf_metrics[1]:.2f} CHF\n" if best_model_name == "Random Forest" else f"RMSE: {gb_metrics[1]:.2f} CHF\n")
    file.write(f"R²: {rf_metrics[2]:.4f}\n" if best_model_name == "Random Forest" else f"R²: {gb_metrics[2]:.4f}\n")
    file.write(f"CV RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f} CHF\n" if best_model_name == "Random Forest" else f"CV RMSE: {np.sqrt(-gb_cv_scores.mean()):.2f} CHF\n") # Kreuzvalidierungs-Ergebnisse einbeziehen (negative Werte werden in positive umgerechnet)

# Versucht, den Ordner zu öffnen
try:
    os.startfile(models_dir)
except:
    pass
