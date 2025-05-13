import pandas as pd # data manipulation library
import numpy as np # data manipulation library
import pickle # for saving trained models to files
import os # file system operations
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # machine learning models and utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Again we define the same output directory as done before in data_preparation.py and generate_travel_times.py
output_dir = r"C:\Users\rinor\OneDrive\Desktop\Computer Science Project\Data"
processed_dir = os.path.join(output_dir, "processed") # to store intermediate data files
models_dir = os.path.join(output_dir, "models") # to store trained ML models and related files
os.makedirs(processed_dir, exist_ok=True) # Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True) # Create directories if they don't exist

# Again we use the direct GitHub URL to read the CSV file
url_modell_input_final = 'https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv'
try: # attempt to load data from Git Repo first
    df = pd.read_csv('https://raw.githubusercontent.com/Rinor909/zurich-real-estate/refs/heads/main/data/processed/modell_input_final.csv')
except: # fall back to local file if git repo fails
    df = pd.read_csv(os.path.join(processed_dir, 'modell_input_final.csv'))

# Feature und Zielwerte definieren
# Features: Quartier, Zimmeranzahl, Preisniveau, Baujahr, etc.
# Ziel: MedianPreis
X = df.drop(['MedianPreis', 'Quartier', 'Zimmeranzahl'], axis=1) # contains all features for prediction # drops the target variable and redundant columns
y = df['MedianPreis'] # target variable

# Kategoriale und numerische Features identifizieren # necessary separation for proper processing in sklearn pipelines
cat_features = ['Quartier_Code'] # categorical since it is a neighborhood identifier
num_features = [col for col in X.columns if col not in cat_features] # all else numerical

# Train-Test-Split (80/20) # training data is used to build the model # test data is held out to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% data # random_state = 42 ensures reproducible results

# Modell-Pipeline erstellen mit One-Hot-Encoding für kategoriale Features # handles different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features), # numerical features passthrough unchanged
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features) # one hot-encoded (converted to binary columns) # handle_unknown used to handle new categories at prediction time
    ])

# Due to the course's limited explanation of random forests and gradient boosting regressors, I had to code this part with AI assistance
# Define models
# Random Forest Regressor # an ensemble of decision trees trained on random subsets of data
rf_model = Pipeline([
    ('preprocessor', preprocessor), # each pipeline integrates the preprocessor with the regressor
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)) # n_estimators sets the number of trees in each ensemble # random_state ensures reproducible results
])

# Gradient Boosting Regressor # an ensemble method that builds trees sequentially
gb_model = Pipeline([
    ('preprocessor', preprocessor), # each pipeline integrates the preprocessor with the regressor
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42)) # n_estimators sets the number of trees in each ensemble # random_state ensures reproducible results
])

# Modelle trainieren # models learn patterns that relate features to property prices # most intensive part of the code
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred) # Average absolute difference between predictions and actual values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Square root of average squared differences
    r2 = r2_score(y_test, y_pred) # Proportion of variance explained by the model
    return mae, rmse, r2

# Evalution of models by appliying function to both # storing performance metrics for comparison
rf_metrics = evaluate_model(rf_model, X_test, y_test) 
gb_metrics = evaluate_model(gb_model, X_test, y_test)

# Cross-Validation durchführen # it tests model on multiple train/test splits to ensure robustness
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error') # 5-fold cross-validation for both models # MSE used as scoring metric
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error') # returns negative values which will be converted to positive later

# Feature Importance für Random Forest Modell
if hasattr(rf_model.named_steps['regressor'], 'feature_importances_'): # extracts feature importance from Random Forest model
    # Column names nach der Transformation durch OneHotEncoder
    ohe = preprocessor.named_transformers_['cat'] 
    cat_cols = ohe.get_feature_names_out(['Quartier_Code'])
    feature_names = num_features + list(cat_cols) # gets transformed feature names after one-hot encoding 
    # Feature Importance extrahieren
    importances = rf_model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1] # sorts features by importance (descending) which helps identify which factors influence property prices the most

# Beste Modell auswählen
best_model = rf_model if rf_metrics[2] > gb_metrics[2] else gb_model # compares models based on R^2 performance (rf_metrics[2])
best_model_name = "Random Forest" if rf_metrics[2] > gb_metrics[2] else "Gradient Boosting" # choose the model with the highest R^2 performance as best model # records name of best model for documentation

# Modell speichern 
with open(os.path.join(models_dir, 'price_model.pkl'), 'wb') as file: # wb opens the file in binary white mode
    pickle.dump(best_model, file) # save the best model to a pickle file which allows us to reuse the model later without retraining

# Quartier-Mapping für die Anwendung speichern
quartier_mapping = {code: quartier for code, quartier in zip(df['Quartier_Code'], df['Quartier'])} # creating a dictionary that maps numeric codes to neighborhood names
with open(os.path.join(models_dir, 'quartier_mapping.pkl'), 'wb') as file: 
    pickle.dump(quartier_mapping, file) # save this mapping to a pickle file # helps translate model outputs back to human-readable neighborhood names

# Save model metrics
with open(os.path.join(models_dir, 'model_metrics.txt'), 'w') as file: # saves performance metrics to .txt file
    file.write(f"Model: {best_model_name}\n") # says which model was selected and then its performance metrics below # formatting of values with appropriate precision
    file.write(f"MAE: {rf_metrics[0]:.2f} CHF\n" if best_model_name == "Random Forest" else f"MAE: {gb_metrics[0]:.2f} CHF\n")
    file.write(f"RMSE: {rf_metrics[1]:.2f} CHF\n" if best_model_name == "Random Forest" else f"RMSE: {gb_metrics[1]:.2f} CHF\n")
    file.write(f"R²: {rf_metrics[2]:.4f}\n" if best_model_name == "Random Forest" else f"R²: {gb_metrics[2]:.4f}\n")
    file.write(f"CV RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f} CHF\n" if best_model_name == "Random Forest" else f"CV RMSE: {np.sqrt(-gb_cv_scores.mean()):.2f} CHF\n") # we include cross-validation resutls (converting negative values to positive)

# Try to open the folder
try:
    os.startfile(models_dir)
except:
    pass