# Zurich Real Estate Price Prediction

This project predicts real estate prices in Zurich based on property characteristics (location, building age, room count) and travel time to key destinations using machine learning models.

## Project Overview

The application enables users to:
- Predict property prices based on selected neighborhood, room count, and building age
- Visualize price distributions across Zurich neighborhoods using interactive maps
- Analyze how travel time to key destinations affects property prices
- Explore historical price trends by neighborhood and room count

## Data Sources

The project uses the following datasets:
- `bau515od5155.csv`: Property prices by neighborhood (2009-2024)
- `bau515od5156.csv`: Property prices by building age (2009-2024)
- Generated travel time data to key Zurich destinations

## Project Structure

```
project/
├── app/                     # Streamlit application files
│   ├── app.py               # Main application entry point
│   ├── maps.py              # Map visualization functions
│   └── utils.py             # Helper utilities
├── data/                    # Data directory
│   ├── raw/                 # Raw datasets
│   │   ├── bau515od5155.csv # Neighborhood price data
│   │   └── bau515od5156.csv # Building age price data
│   └── processed/           # Cleaned and processed datasets
├── models/                  # Trained ML models
├── notebooks/               # Jupyter notebooks
│   ├── eda.ipynb            # Exploratory data analysis
│   └── model_dev.ipynb      # Model development
├── scripts/                 # Processing scripts
│   ├── data_preparation.py  # Data cleaning script
│   ├── generate_travel_times.py # Script to generate travel time data
│   └── model_training.py    # Model training script
├── tests/                   # Test scripts
│   └── test_pipeline.py     # Data pipeline tests
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

## Installation and Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Process the raw data:
   ```bash
   python scripts/data_preparation.py
   ```
4. Generate travel time data:
   ```bash
   python scripts/generate_travel_times.py
   ```
5. Train the prediction model:
   ```bash
   python scripts/model_training.py
   ```
6. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

After launching the application:

1. Use the sidebar to select:
   - Property neighborhood
   - Number of rooms
   - Building age
   - Maximum travel time

2. View the predicted price and visualizations including:
   - Price heatmap across Zurich neighborhoods
   - Travel time analysis to key destinations
   - Historical price trends

## Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models (RandomForest, GradientBoosting)
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Static visualizations

## Project Development

The project follows this development workflow:

1. **Data Preparation (Apr 29 – May 3)**
   - Clean and merge datasets
   - Generate travel time data
   - Conduct exploratory data analysis
   - Create initial visualizations

2. **Model Development (May 4 – May 8)**
   - Train baseline ML models
   - Build and validate hybrid model
   - Tune model performance

3. **Application Development (May 9 – May 12)**
   - Build Streamlit UI
   - Implement interactive Zurich maps
   - Create filtering features

4. **Testing & Finalization (May 13 – May 15)**
   - Test app and model
   - Finalize presentation
   - Submit project

## Contributors

- Rinor: ML model, API integration, backend
- Matteo: Streamlit UI, frontend development
- Matthieu: Visualizations, Zurich maps
- Anna: Testing, documentation, support

## License

This project is intended for educational purposes.

## Acknowledgments

- Zurich Open Data platform for providing the property price datasets
