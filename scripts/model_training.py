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

# Daten laden
df = pd.read_csv('data/processed/modell_input_final.csv')

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
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/price_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Modell wurde in 'models/price_model.pkl' gespeichert.")

# Quartier-Mapping für die Anwendung speichern
quartier_mapping = {code: quartier for code, quartier in zip(df['Quartier_Code'], df['Quartier'])}
with open('models/quartier_mapping.pkl', 'wb') as file:
    pickle.dump(quartier_mapping, file)

print("Quartier-Mapping wurde in 'models/quartier_mapping.pkl' gespeichert.")