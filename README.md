# 🏡 Zurich Real Estate Price Prediction

A machine learning application that predicts real estate prices in Zurich based on property characteristics (location, building age, room count) and travel time to key destinations.

## Overview

This application helps you:
- Predict property prices in different Zurich neighborhoods
- Visualize price patterns across Zurich on interactive maps
- Analyze how travel time to key destinations affects prices
- Compare prices across neighborhoods and time periods

## Features

- **Machine Learning Prediction**: Estimates property prices using Random Forest and Gradient Boosting models
- **Interactive Maps**: Visualizes price distribution and travel times across Zurich
- **Neighborhood Comparison**: Compares prices and trends across neighborhoods
- **Travel Time Analysis**: Shows how accessibility to key locations affects property values
- **Historical Data**: Analyzes price trends from 2009-2024

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/zurich-real-estate.git
cd zurich-real-estate
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your Google Maps API key for travel time data:
   - Create a `.streamlit/secrets.toml` file with your API key:
   ```toml
   GOOGLE_MAPS_API_KEY = "your-api-key-here"
   ```

## Data Processing

Run the following scripts in sequence to prepare the data:

```bash
# 1. Process the property data
python scripts/data_preparation.py

# 2. Generate travel time data
python scripts/generate_travel_times.py

# 3. Train the prediction model
python scripts/model_training.py
```

## Running the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to use the application.

## Project Structure

```
project/
├── app/
│   ├── app.py           # Streamlit app main file
│   ├── maps.py          # Map visualization functions
│   ├── utils.py         # Helper utilities
├── data/
│   ├── raw/             # Original datasets
│   ├── processed/       # Cleaned and processed data
├── models/              # Trained ML models
├── notebooks/
│   ├── eda.ipynb        # Exploratory Data Analysis
│   ├── model_dev.ipynb  # Model development
├── scripts/
│   ├── data_preparation.py     # Data cleaning scripts
│   ├── generate_travel_times.py # Travel time generation
│   ├── model_training.py       # ML model training
├── .streamlit/          # Streamlit configuration
├── app.py               # Entry point
├── requirements.txt     # Dependencies
├── README.md
```

## Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set your Google Maps API key in the Streamlit Cloud secrets management
5. Configure the main file as `app.py`

## Data Sources

- Property prices by neighborhood (2009-2024)
- Property prices by building age (2009-2024)
- Travel time data to key Zurich destinations (via Google Maps API)

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning models
- **Plotly/Folium**: Interactive visualizations
- **Google Maps API**: Travel time data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
