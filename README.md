# Zurich Real Estate Price Prediction

This project predicts real estate prices in Zurich based on property characteristics and travel time using a machine learning model.  
Built with Streamlit, pandas, scikit-learn, and plotly.

## 🏠 Project Overview

The Zurich Real Estate Price Prediction app helps users predict property prices based on:
- Location (neighborhood)
- Building age
- Room count 
- Travel time to key destinations (Hauptbahnhof, ETH Zurich, Zurich Airport, Bahnhofstrasse)

## 📋 Features

- **Price Prediction**: Get estimated property prices based on input parameters
- **Interactive Maps**: Visualize property prices across Zurich neighborhoods
- **Travel Time Analysis**: See how travel time affects property values
- **Market Insights**: Explore trends in the Zurich real estate market

## 🧰 Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **pandas/numpy**: Data processing and analysis
- **scikit-learn**: Machine learning models (Random Forest, Gradient Boosting)
- **plotly**: Interactive map visualizations
- **Google Maps API**: Travel time data collection

## 📁 Folder Structure

```
project/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Cleaned datasets
├── notebooks/
│   ├── eda.ipynb         # Data exploration
│   ├── model_dev.ipynb   # Model development
├── scripts/
│   ├── data_preparation.py        # Data cleaning
│   ├── generate_travel_times.py   # Google Maps API integration
│   ├── model_training.py          # ML model training
├── app/
│   ├── app.py                     # Streamlit app
│   ├── maps.py                    # Map visualization functions
│   ├── utils.py                   # Helper functions
├── models/
│   ├── price_model.pkl            # Trained model
├── tests/
│   ├── test_pipeline.py           # Basic tests
├── README.md
├── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd zurich-real-estate-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare data:
   ```
   cd scripts
   python data_preparation.py
   python generate_travel_times.py
   python model_training.py
   ```

4. Run the application:
   ```
   cd app
   streamlit run app.py
   ```

## 🔍 Data Sources

- Property Prices by Neighborhood (bau515od5155.csv)
- Property Prices by Building Age (bau515od5156.csv)
- Travel Time Data (generated via Google Maps API)

## 👥 Team Members

- Rinor: ML model, API integration, backend logic
- Matteo: Streamlit UI, frontend development
- Matthieu: Data visualization, Zurich maps
- Anna: Testing, documentation, support

## 📅 Project Timeline

- **Apr 29 - May 3**: Data Preparation & Exploration
- **May 4 - May 8**: Model Development
- **May 9 - May 12**: Application Development
- **May 13 - May 15**: Testing & Finalization

## 📋 TODO

- [ ] Clean raw datasets
- [ ] Generate travel time data
- [ ] Develop baseline models
- [ ] Create interactive Zurich maps
- [ ] Build Streamlit UI
- [ ] Test the application
- [ ] Create video presentation
