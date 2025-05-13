import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Again we define the same output directory as done before in data_preparation.py and generate_travel_times.py
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed")
models_dir = os.path.join(output_dir, "models")
os.makedirs(processed_dir, exist_ok=True) # Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True) # Create directories if they don't exist

# Daten laden
url_modell_input_final = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv'

try:
    df = pd.read_csv('https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv')
except:
    df = pd.read_csv(os.path.join(processed_dir, 'modell_input_final.csv'))

# Load travel times data if exists
travel_times_path = os.path.join(processed_dir, 'travel_times.csv')
if os.path.exists(travel_times_path):
    df_travel_times = pd.read_csv(travel_times_path)
    # Here we could incorporate travel times into the model which would further influence the value
    # This would be an enhancement for a future version or continuation of the project

# Feature und Zielwerte definieren
# Features: Quartier, Zimmeranzahl, Preisniveau, Baujahr, etc.
# Ziel: MedianPreis
X = df.drop(['MedianPreis', 'Quartier', 'Zimmeranzahl'], axis=1)
y = df['MedianPreis']

# Kategoriale und numerische Features identifizieren
cat_features = ['Quartier_Code']
num_features = [col for col in X.columns if col not in cat_features]

# Train-Test-Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell-Pipeline erstellen mit One-Hot-Encoding für kategoriale Features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Define models
# Random Forest Regressor
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Gradient Boosting Regressor
gb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Modelle trainieren
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Evalution of models
rf_metrics = evaluate_model(rf_model, X_test, y_test)
gb_metrics = evaluate_model(gb_model, X_test, y_test)

# Cross-Validation durchführen
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Feature Importance für Random Forest Modell
if hasattr(rf_model.named_steps['regressor'], 'feature_importances_'):
    # Column names nach der Transformation durch OneHotEncoder
    ohe = preprocessor.named_transformers_['cat']
    cat_cols = ohe.get_feature_names_out(['Quartier_Code'])
    feature_names = num_features + list(cat_cols)
    # Feature Importance extrahieren
    importances = rf_model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1]

# Beste Modell auswählen
best_model = rf_model if rf_metrics[2] > gb_metrics[2] else gb_model
best_model_name = "Random Forest" if rf_metrics[2] > gb_metrics[2] else "Gradient Boosting"

# Modell speichern 
with open(os.path.join(models_dir, 'price_model.pkl'), 'wb') as file:
    pickle.dump(best_model, file)

# Quartier-Mapping für die Anwendung speichern
quartier_mapping = {code: quartier for code, quartier in zip(df['Quartier_Code'], df['Quartier'])}
with open(os.path.join(models_dir, 'quartier_mapping.pkl'), 'wb') as file:
    pickle.dump(quartier_mapping, file)

# Save model metrics
with open(os.path.join(models_dir, 'model_metrics.txt'), 'w') as file:
    file.write(f"Model: {best_model_name}\n")
    file.write(f"MAE: {rf_metrics[0]:.2f} CHF\n" if best_model_name == "Random Forest" else f"MAE: {gb_metrics[0]:.2f} CHF\n")
    file.write(f"RMSE: {rf_metrics[1]:.2f} CHF\n" if best_model_name == "Random Forest" else f"RMSE: {gb_metrics[1]:.2f} CHF\n")
    file.write(f"R²: {rf_metrics[2]:.4f}\n" if best_model_name == "Random Forest" else f"R²: {gb_metrics[2]:.4f}\n")
    file.write(f"CV RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f} CHF\n" if best_model_name == "Random Forest" else f"CV RMSE: {np.sqrt(-gb_cv_scores.mean()):.2f} CHF\n")

# Try to open the folder
try:
    os.startfile(models_dir)
except:
    pass