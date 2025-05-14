# 🏡 Zurich Real Estate Price Prediction

A web application that shows real estate price estimates in Zurich based on location, number of rooms, and building age.

## What This Project Does

This app helps you:
- See estimated property prices in different Zurich neighborhoods
- View price differences across Zurich on a map
- Check how travel time affects prices
- Compare prices between neighborhoods

## Setup Instructions

1. Clone this repository
```
git clone https://github.com/yourusername/zurich-real-estate.git
cd zurich-real-estate
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Add your Google Maps API key
   - Create a `.streamlit/secrets.toml` file
   - Add this line with your API key: `GOOGLE_MAPS_API_KEY = "your-api-key-here"`

## Preparing the Data

Run these scripts in order:

```
# 1. Process the property data
python scripts/data_preparation.py

# 2. Generate travel time data
python scripts/generate_travel_times.py

# 3. Train the prediction model
python scripts/model_training.py
```

## Running the App

```
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`

## Project Files

- `app.py`: Main application file
- `app/`: Directory with app components
- `data/`: Contains raw and processed data
- `models/`: Stores the trained ML model
- `scripts/`: Contains data processing scripts
- `requirements.txt`: Lists all required packages

## Features

- **Price Prediction**: Estimates property prices using a machine learning model
- **Interactive Maps**: Shows price distribution across Zurich
- **Travel Time Analysis**: Displays how travel times affect property values
- **Price Comparison**: Compares prices between neighborhoods

## Data Sources

- Property prices by neighborhood (2009-2024)
- Property prices by building age (2009-2024)
- Travel times to key locations (calculated using Google Maps API)

## Tools Used

- **Python**: Programming language
- **Streamlit**: Web interface
- **Pandas**: Data processing
- **Scikit-learn**: Machine learning for predictions
- **Plotly**: Interactive charts and maps
- **Google Maps API**: Travel time data

## Note

This project was created for a Computer Science course. The predictions are estimates only and should not be used for actual real estate decisions.
