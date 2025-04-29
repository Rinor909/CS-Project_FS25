# Zurich Real Estate Price Prediction - Setup Guide

This guide will walk you through setting up and running the Zurich Real Estate Price Prediction project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Step 1: Set up the project structure

Run the setup script to create the necessary directory structure:

```bash
chmod +x create_folders.sh
./create_folders.sh
```

## Step 2: Install dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Step 3: Place data files

Place the CSV files in the correct location:

1. Move `bau515od5155.csv` (neighborhood price data) to `data/raw/`
2. Move `bau515od5156.csv` (building age price data) to `data/raw/`

## Step 4: Process the data

Run the data preparation script to process the raw data:

```bash
python scripts/data_preparation.py
```

You should see output indicating that the data has been processed and saved to the `data/processed/` directory.

## Step 5: Generate travel times

Run the script to generate travel time data:

```bash
python scripts/generate_travel_times.py
```

This will create a `travel_times.json` file in the `data/processed/` directory.

## Step 6: Train the model

Train the machine learning model using the processed data:

```bash
python scripts/model_training.py
```

The trained model will be saved to the `models/` directory as `price_model.pkl`.

## Step 7: Run the application

Launch the Streamlit app:

```bash
streamlit run app.py
```

This will start the web application, which should be accessible at http://localhost:8501 in your web browser.

## Step 8: Explore the application

Use the sidebar to set different parameters:
- Select a neighborhood
- Choose room count
- Select building age
- Set maximum travel time
- Select key destinations

The application will display:
- Predicted property price
- Price distribution map
- Travel time analysis
- Historical price trends

## Troubleshooting

If you encounter any issues:

1. Check that all files are in the correct locations
2. Verify that all dependencies are installed
3. Look for error messages in the terminal
4. Ensure the CSV files are properly formatted

If the application can't find the data files, make sure they are in the `data/raw/` directory and have the correct names.

## Running tests

To verify that the data pipeline is working correctly, run the tests:

```bash
python -m unittest tests/test_pipeline.py
```

## Exploring the data

To explore the data in more detail, you can use the Jupyter notebooks:

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model_dev.ipynb
```

These notebooks provide detailed analysis of the property price data and model development process.
