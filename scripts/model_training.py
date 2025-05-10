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

# Define the same output directory as in your data preparation script
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed")
models_dir = os.path.join(output_dir, "models")

# Create directories if they don't exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print(f"Output directory: {output_dir}")
print(f"Models will be saved to: {models_dir}")

# Daten laden
url_modell_input_final = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv'
print("Loading model input data from GitHub...")

try:
    df = pd.read_csv(url_modell_input_final)
    print(f"Loaded {len(df)} rows from GitHub.")
except Exception as e:
    # Try local file as fallback
    print(f"Error loading from GitHub: {e}")
    local_path = os.path.join(processed_dir, 'modell_input_final.csv')
    print(f"Trying to load from local path: {local_path}")
    df = pd.read_csv(local_path)
    print(f"Loaded {len(df)} rows from local file.")

# Load travel times data if it exists
travel_times_path = os.path.join(processed_dir, 'travel_times.csv')
if os.path.exists(travel_times_path):
    print("Loading travel times data...")
    df_travel_times = pd.read_csv(travel_times_path)
    print(f"Loaded {len(df_travel_times)} travel time records.")
    
    # Here you could incorporate travel times into your model if desired
    # This would be an enhancement for a future version

# Feature und Zielwerte definieren
# Features: Quartier, Zimmeranzahl, Preisniveau, Baujahr, etc.
# Ziel: MedianPreis
X = df.drop(['MedianPreis', 'Quartier', 'Zimmeranzahl'], axis=1)
y = df['MedianPreis']

print(f"Features: {X.columns.tolist()}")
print(f"Number of features: {len(X.columns)}")
print(f"Number of samples: {len(X)}")

# Kategoriale und numerische Features identifizieren
cat_features = ['Quartier_Code']
num_features = [col for col in X.columns if col not in cat_features]

# Train-Test-Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Modell-Pipeline erstellen mit One-Hot-Encoding für kategoriale Features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

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
print("Random Forest Modell wird trainiert...")
rf_model.fit(X_train, y_train)

print("Gradient Boosting Modell wird trainiert...")
gb_model.fit(X_train, y_train)

# Modelle evaluieren
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f} CHF")
    print(f"RMSE: {rmse:.2f} CHF")
    print(f"R²: {r2:.4f}")
    
    return mae, rmse, r2

print("\nRandom Forest Modell Evaluation:")
rf_metrics = evaluate_model(rf_model, X_test, y_test)

print("\nGradient Boosting Modell Evaluation:")
gb_metrics = evaluate_model(gb_model, X_test, y_test)

# Cross-Validation durchführen
print("\nCross-Validation (5-fold) wird durchgeführt...")
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')

print(f"Random Forest CV RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f} CHF")
print(f"Gradient Boosting CV RMSE: {np.sqrt(-gb_cv_scores.mean()):.2f} CHF")

# Feature Importance für Random Forest Modell
if hasattr(rf_model.named_steps['regressor'], 'feature_importances_'):
    # Column names nach der Transformation durch OneHotEncoder
    ohe = preprocessor.named_transformers_['cat']
    cat_cols = ohe.get_feature_names_out(['Quartier_Code'])
    feature_names = num_features + list(cat_cols)
    
    # Feature Importance extrahieren
    importances = rf_model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance (Random Forest):")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Beste Modell auswählen
best_model = rf_model if rf_metrics[2] > gb_metrics[2] else gb_model
best_model_name = "Random Forest" if rf_metrics[2] > gb_metrics[2] else "Gradient Boosting"
print(f"\nBestes Modell basierend auf R²: {best_model_name}")

# Modell speichern
model_path = os.path.join(models_dir, 'price_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Modell wurde in '{model_path}' gespeichert.")

# Quartier-Mapping für die Anwendung speichern
quartier_mapping = {code: quartier for code, quartier in zip(df['Quartier_Code'], df['Quartier'])}
mapping_path = os.path.join(models_dir, 'quartier_mapping.pkl')
with open(mapping_path, 'wb') as file:
    pickle.dump(quartier_mapping, file)

print(f"Quartier-Mapping wurde in '{mapping_path}' gespeichert.")

# Save the model evaluation metrics for reference
metrics_path = os.path.join(models_dir, 'model_metrics.txt')
with open(metrics_path, 'w') as file:
    file.write(f"Model: {best_model_name}\n")
    file.write(f"MAE: {rf_metrics[0]:.2f} CHF\n" if best_model_name == "Random Forest" else f"MAE: {gb_metrics[0]:.2f} CHF\n")
    file.write(f"RMSE: {rf_metrics[1]:.2f} CHF\n" if best_model_name == "Random Forest" else f"RMSE: {gb_metrics[1]:.2f} CHF\n")
    file.write(f"R²: {rf_metrics[2]:.4f}\n" if best_model_name == "Random Forest" else f"R²: {gb_metrics[2]:.4f}\n")
    file.write(f"CV RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f} CHF\n" if best_model_name == "Random Forest" else f"CV RMSE: {np.sqrt(-gb_cv_scores.mean()):.2f} CHF\n")
    
print(f"Model metrics saved to '{metrics_path}'")

# Print summary
print("\n=== Training Summary ===")
print(f"Best model: {best_model_name}")
print(f"Performance (R²): {rf_metrics[2]:.4f}" if best_model_name == "Random Forest" else f"Performance (R²): {gb_metrics[2]:.4f}")
print(f"Files saved to: {models_dir}")

# Try to open the folder in Windows Explorer
try:
    os.startfile(models_dir)
    print("Opening folder in Windows Explorer...")
except Exception as e:
    print(f"Could not open folder: {e}")