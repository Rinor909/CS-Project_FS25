import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_models_dir():
    """Create models directory if it doesn't exist"""
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def load_processed_data():
    """Load processed data for model training"""
    processed_dir = os.path.join('data', 'processed')
    
    # Load neighborhood data
    neighborhood_path = os.path.join(processed_dir, 'processed_neighborhood_data.csv')
    if not os.path.exists(neighborhood_path):
        raise FileNotFoundError(f"File not found: {neighborhood_path}")
    
    neighborhood_df = pd.read_csv(neighborhood_path)
    
    # Load building age data
    building_age_path = os.path.join(processed_dir, 'processed_building_age_data.csv')
    if not os.path.exists(building_age_path):
        raise FileNotFoundError(f"File not found: {building_age_path}")
    
    building_age_df = pd.read_csv(building_age_path)
    
    print(f"Loaded processed data: {len(neighborhood_df)} neighborhood records, {len(building_age_df)} building age records")
    
    return neighborhood_df, building_age_df

def prepare_features(neighborhood_df, building_age_df):
    """Prepare features for model training"""
    # For this simplified version, we'll use only the neighborhood data
    # In a real implementation, we would combine both datasets properly
    
    # Use the latest year data
    latest_year = neighborhood_df['year'].max()
    latest_data = neighborhood_df[neighborhood_df['year'] == latest_year].copy()
    
    # Select features
    X = latest_data[['neighborhood', 'room_count']]
    y = latest_data['median_price']
    
    print(f"Prepared features using {len(X)} records from {latest_year}")
    
    return X, y, latest_year

def train_model(X, y):
    """Train a Random Forest model on the data"""
    # Define categorical features
    categorical_features = ['neighborhood', 'room_count']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create model pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation:")
    print(f"MAE: {mae:.2f} CHF")
    print(f"RMSE: {rmse:.2f} CHF")
    print(f"R²: {r2:.4f}")
    
    # Also train a Gradient Boosting model
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
    
    print("Training Gradient Boosting model...")
    gb_pipeline.fit(X_train, y_train)
    
    # Evaluate the GB model
    y_pred_gb = gb_pipeline.predict(X_test)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    
    print(f"Gradient Boosting model evaluation:")
    print(f"MAE: {mae_gb:.2f} CHF")
    print(f"RMSE: {rmse_gb:.2f} CHF")
    print(f"R²: {r2_gb:.4f}")
    
    # Choose the better model
    if r2_gb > r2:
        print("Gradient Boosting model performed better. Saving this model.")
        return gb_pipeline
    else:
        print("Random Forest model performed better. Saving this model.")
        return rf_pipeline

def save_model(model, models_dir, year):
    """Save the trained model to disk"""
    model_path = os.path.join(models_dir, 'price_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Save model metadata
    metadata = {
        'year': year,
        'features': ['neighborhood', 'room_count'],
        'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    metadata_path = os.path.join(models_dir, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model metadata saved to {metadata_path}")

def main():
    """Main function to train and save the model"""
    print("Starting model training...")
    
    # Create models directory
    models_dir = create_models_dir()
    
    try:
        # Load processed data
        neighborhood_df, building_age_df = load_processed_data()
        
        # Prepare features
        X, y, latest_year = prepare_features(neighborhood_df, building_age_df)
        
        # Train model
        model = train_model(X, y)
        
        # Save model
        save_model(model, models_dir, latest_year)
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()