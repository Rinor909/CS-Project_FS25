"""
Model Training Script for Zurich Real Estate Price Prediction
------------------------------------------------------------
Purpose: Train and evaluate ML models for real estate price prediction

Tasks:
1. Load processed datasets
2. Prepare features and target variables
3. Train baseline models (Random Forest and Gradient Boosting)
4. Evaluate models (MAE, RMSE, R²)
5. Save trained models for use in the Streamlit app

Owner: Rinor (Primary), Matthieu (Support)
"""

import pandas as pd
import numpy as np
import os
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
PROCESSED_DATA_DIR = "../data/processed"
MODELS_DIR = "../models"
NEIGHBORHOOD_DATA = "processed_neighborhood.csv"
BUILDING_AGE_DATA = "processed_building_age.csv"
TRAVEL_TIME_DATA = "neighborhood_travel_times.csv"
MODEL_OUTPUT = "price_model.pkl"
FEATURE_IMPORTANCE_FIG = "feature_importance.png"

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {MODELS_DIR}")

def load_datasets():
    """Load processed datasets."""
    try:
        # Load neighborhood data
        neighborhood_path = os.path.join(PROCESSED_DATA_DIR, NEIGHBORHOOD_DATA)
        neighborhood_df = pd.read_csv(neighborhood_path)
        logger.info(f"Loaded neighborhood data: {neighborhood_path}")
        
        # Load building age data
        building_age_path = os.path.join(PROCESSED_DATA_DIR, BUILDING_AGE_DATA)
        building_age_df = pd.read_csv(building_age_path)
        logger.info(f"Loaded building age data: {building_age_path}")
        
        # Load travel time data
        travel_time_path = os.path.join(PROCESSED_DATA_DIR, TRAVEL_TIME_DATA)
        travel_time_df = pd.read_csv(travel_time_path)
        logger.info(f"Loaded travel time data: {travel_time_path}")
        
        return neighborhood_df, building_age_df, travel_time_df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def prepare_model_data(neighborhood_df, building_age_df, travel_time_df):
    """
    Prepare data for model training.
    
    TODO:
    - Merge datasets
    - Create features (one-hot encoding for categorical variables)
    - Split into features (X) and target (y)
    """
    logger.info("Preparing data for model training")
    
    # This is a placeholder - implement actual data preparation
    # For example:
    # 1. Convert neighborhood to one-hot encoding
    # 2. Combine with building age and travel time features
    # 3. Create any additional features
    
    # Dummy implementation - replace with actual implementation
    # For now, we'll just create a dummy dataset
    np.random.seed(42)
    X = pd.DataFrame({
        'neighborhood_factor': np.random.random(100),
        'building_age': np.random.randint(0, 100, 100),
        'room_count': np.random.randint(1, 6, 100),
        'travel_time_hauptbahnhof': np.random.randint(5, 45, 100),
        'travel_time_eth': np.random.randint(10, 50, 100),
        'travel_time_airport': np.random.randint(20, 80, 100),
        'travel_time_bahnhofstrasse': np.random.randint(5, 45, 100)
    })
    
    # Generate target variable (price) with some relationship to features
    y = (
        500000 +
        X['neighborhood_factor'] * 1000000 -
        X['building_age'] * 2000 +
        X['room_count'] * 150000 -
        X['travel_time_hauptbahnhof'] * 5000 -
        X['travel_time_airport'] * 1000 +
        np.random.normal(0, 100000, 100)  # Add some noise
    )
    
    logger.info(f"Prepared dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples)")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    logger.info("Training Random Forest model")
    
    # Define model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    logger.info("Random Forest model trained")
    return model

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model."""
    logger.info("Training Gradient Boosting model")
    
    # Define model parameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # Train model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    logger.info("Gradient Boosting model trained")
    return model

def perform_grid_search(X_train, y_train, model_type='rf'):
    """
    Perform grid search to find optimal hyperparameters.
    
    TODO:
    - Implement grid search for both Random Forest and Gradient Boosting
    - Return best model
    """
    logger.info(f"Performing grid search for {model_type}")
    
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:  # Gradient Boosting
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    # This is commented out because it's computationally expensive
    # Uncomment when you're ready to run grid search
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {-grid_search.best_score_}")
    
    return grid_search.best_estimator_
    """
    
    logger.info("Grid search placeholder - returning default model")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model evaluation results:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  R²: {r2:.4f}")
    
    return mae, rmse, r2

def cross_validate(model, X, y, cv=5):
    """Perform cross-validation."""
    scores = cross_val_score(
        model, X, y, cv=cv,
        scoring='neg_mean_squared_error'
    )
    rmse_scores = np.sqrt(-scores)
    
    logger.info(f"Cross-validation RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f})")
    
    return rmse_scores

def plot_feature_importance(model, X):
    """Plot feature importance."""
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    output_path = os.path.join(MODELS_DIR, FEATURE_IMPORTANCE_FIG)
    plt.savefig(output_path)
    logger.info(f"Feature importance plot saved to {output_path}")
    
    return feature_importance

def save_model(model, X):
    """Save trained model to disk."""
    output_path = os.path.join(MODELS_DIR, MODEL_OUTPUT)
    
    # Save model and feature names
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': X.columns.tolist()
        }, f)
    
    logger.info(f"Model saved to {output_path}")

def main():
    """Main model training pipeline."""
    start_time = datetime.now()
    logger.info("Starting model training pipeline")
    
    # Create directories
    create_directories()
    
    try:
        # Load datasets
        neighborhood_df, building_age_df, travel_time_df = load_datasets()
        
        # Prepare data
        X, y = prepare_model_data(neighborhood_df, building_age_df, travel_time_df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train models
        rf_model = train_random_forest(X_train, y_train)
        gb_model = train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        logger.info("Random Forest model evaluation:")
        rf_mae, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
        
        logger.info("Gradient Boosting model evaluation:")
        gb_mae, gb_rmse, gb_r2 = evaluate_model(gb_model, X_test, y_test)
        
        # Choose best model
        if rf_r2 > gb_r2:
            best_model = rf_model
            logger.info("Random Forest selected as best model")
        else:
            best_model = gb_model
            logger.info("Gradient Boosting selected as best model")
        
        # Cross-validate best model
        cross_validate(best_model, X, y)
        
        # Plot feature importance
        plot_feature_importance(best_model, X)
        
        # Save model
        save_model(best_model, X)
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise
    
    end_time = datetime.now()
    logger.info(f"Model training pipeline completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
