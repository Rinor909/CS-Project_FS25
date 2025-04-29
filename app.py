import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(
    page_title="Zurich Real Estate Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# Create directory structure if it doesn't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create maps.py if it doesn't exist
if not os.path.exists("maps.py"):
    with open("maps.py", "w") as f:
        # You'd need to paste the maps.py content here
        # For brevity, I'll just create a placeholder that will be replaced
        f.write('# maps.py will be placed here')

# Import the maps module (will be created if not exists)
try:
    from maps import display_map, predict_prices_for_all_neighborhoods
    maps_available = True
except ImportError:
    maps_available = False
    st.warning("Maps functionality not available. Please make sure maps.py is in the current directory.")

# Title
st.title("🏡 Zurich Real Estate Price Prediction")
st.write("This app predicts real estate prices in Zurich based on property characteristics and travel time. Select parameters on the left sidebar to get price predictions and view visualizations.")

# Move CSV files to correct location if they exist in current directory
csv_files = ["bau515od5155.csv", "bau515od5156.csv"]
for csv_file in csv_files:
    if os.path.exists(csv_file) and not os.path.exists(f"data/raw/{csv_file}"):
        os.makedirs("data/raw", exist_ok=True)
        import shutil
        shutil.copy(csv_file, f"data/raw/{csv_file}")
        st.success(f"Moved {csv_file} to data/raw/ directory")

# Function to load data
@st.cache_data
def load_data():
    try:
        # Check if CSV files exist in raw directory
        if not os.path.exists("data/raw/bau515od5155.csv"):
            st.error("Error loading data: File not found: data/raw/bau515od5155.csv")
            st.info("Please place the CSV files in the data/raw directory or upload them below")
            return None, None
        
        if not os.path.exists("data/raw/bau515od5156.csv"):
            st.error("Error loading data: File not found: data/raw/bau515od5156.csv")
            st.info("Please place the CSV files in the data/raw directory or upload them below")
            return None, None
            
        # Load neighborhood data
        neighborhood_data = pd.read_csv("data/raw/bau515od5155.csv", delimiter=",")
        
        # Load building age data
        building_age_data = pd.read_csv("data/raw/bau515od5156.csv", delimiter=",")
        
        return neighborhood_data, building_age_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Allow file upload
with st.expander("Upload Data Files"):
    st.write("Upload the CSV files if they're not already in the data/raw directory")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file1 = st.file_uploader("Upload neighborhood data (bau515od5155.csv)", type="csv")
        if uploaded_file1 is not None:
            os.makedirs("data/raw", exist_ok=True)
            with open(os.path.join("data/raw", "bau515od5155.csv"), "wb") as f:
                f.write(uploaded_file1.getbuffer())
            st.success("File uploaded successfully!")
    
    with col2:
        uploaded_file2 = st.file_uploader("Upload building age data (bau515od5156.csv)", type="csv")
        if uploaded_file2 is not None:
            os.makedirs("data/raw", exist_ok=True)
            with open(os.path.join("data/raw", "bau515od5156.csv"), "wb") as f:
                f.write(uploaded_file2.getbuffer())
            st.success("File uploaded successfully!")

# Function to generate and save travel time data
def generate_travel_times():
    # Sample neighborhoods
    neighborhoods = ["Altstadt", "Escher Wyss", "Gewerbeschule", "Hochschulen", "Höngg", "Oerlikon", 
                     "Seebach", "Altstetten", "Albisrieden"]
    
    # Sample destinations
    destinations = ["Hauptbahnhof", "ETH Zurich", "Zurich Airport", "Bahnhofstrasse"]
    
    # Create sample travel time data
    travel_times = {}
    for neighborhood in neighborhoods:
        travel_times[neighborhood] = {}
        
        # Generate times based on rough geographic knowledge of Zurich
        if neighborhood in ["Altstadt", "Hochschulen"]:
            # Central neighborhoods
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(5, 15)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(5, 15)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(25, 40)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(5, 15)
        elif neighborhood in ["Oerlikon", "Seebach"]:
            # Northern neighborhoods (closer to airport)
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(15, 25)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(15, 25)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(10, 20)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(20, 30)
        else:
            # Other neighborhoods
            travel_times[neighborhood]["Hauptbahnhof"] = np.random.randint(15, 30)
            travel_times[neighborhood]["ETH Zurich"] = np.random.randint(20, 35)
            travel_times[neighborhood]["Zurich Airport"] = np.random.randint(30, 50)
            travel_times[neighborhood]["Bahnhofstrasse"] = np.random.randint(20, 40)
    
    # Save travel time data
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/travel_times.json", "w") as f:
        json.dump(travel_times, f)
    
    return travel_times

# Function to load or create travel time data
def load_travel_time_data():
    try:
        if os.path.exists("data/processed/travel_times.json"):
            with open("data/processed/travel_times.json", "r") as f:
                return json.load(f)
        else:
            st.warning("Travel time data not available: Creating sample travel time data...")
            return generate_travel_times()
    except Exception as e:
        st.error(f"Error loading travel time data: {e}")
        return None

# Function to train and save a simple model
def train_simple_model(neighborhood_data=None, building_age_data=None):
    try:
        # Create a simple random forest model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create synthetic training data if real data isn't available
        X = np.random.rand(100, 4)  # [neighborhood_code, rooms, building_age, travel_time]
        y = 1000000 + 500000 * X[:, 0] + 200000 * X[:, 1] - 10000 * X[:, 2] - 5000 * X[:, 3]
        
        # Train the model
        model.fit(X, y)
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        with open("models/price_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Function to load or create price prediction model
def load_model():
    try:
        if os.path.exists("models/price_model.pkl"):
            with open("models/price_model.pkl", "rb") as f:
                return pickle.load(f)
        else:
            st.warning("Price prediction model not available: Creating sample model...")
            return train_simple_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to predict price
def predict_price(neighborhood, room_count, building_age, max_travel_time):
    # This is a placeholder function
    # In a real app, this would use the trained model
    base_price = 1000000  # Base price in CHF
    
    # Neighborhood adjustment (would come from real data)
    neighborhood_factors = {
        "Altstadt": 1.5,
        "Escher Wyss": 1.3,
        "Gewerbeschule": 1.2,
        "Hochschulen": 1.4,
        "Höngg": 1.1,
        "Oerlikon": 1.0,
        "Seebach": 0.9,
        "Altstetten": 0.95,
        "Albisrieden": 0.97,
        "Sihlfeld": 1.05,
        "Friesenberg": 1.15,
        "Leimbach": 0.95,
        "Wollishofen": 1.2,
        "Enge": 1.4,
        "Wiedikon": 1.1,
        "Hard": 1.0,
        "Unterstrass": 1.2,
        "Oberstrass": 1.25
    }
    
    neighborhood_factor = neighborhood_factors.get(neighborhood, 1.0)
    
    # Room count adjustment
    room_factor = 0.2 * room_count
    
    # Building age adjustment (newer buildings are more expensive)
    age_factor = 1.0
    if building_age == "before 1919":
        age_factor = 0.9
    elif building_age == "1919-1945":
        age_factor = 0.95
    elif building_age == "1946-1970":
        age_factor = 1.0
    elif building_age == "1971-1990":
        age_factor = 1.05
    elif building_age == "1991-2005":
        age_factor = 1.1
    elif building_age == "after 2005":
        age_factor = 1.15
    
    # Travel time adjustment
    travel_factor = 1.0 - (max_travel_time / 100)
    
    # Calculate price
    price = base_price * neighborhood_factor * (1 + room_factor) * age_factor * travel_factor
    
    return price

# Generate files if they don't exist
if not os.path.exists("data/processed/travel_times.json"):
    travel_times = generate_travel_times()
    st.success("Created travel times data!")
else:
    travel_times = load_travel_time_data()

if not os.path.exists("models/price_model.pkl"):
    model = train_simple_model()
    st.success("Created price prediction model!")
else:
    model = load_model()

# Load data
neighborhood_data, building_age_data = load_data()

# Setup dependencies for maps
try:
    # Check if required packages are installed
    import folium
    from streamlit_folium import folium_static
    maps_dependencies_installed = True
except ImportError:
    st.warning("Map visualization dependencies not installed. Installing now...")
    import subprocess
    subprocess.run(["pip", "install", "folium", "streamlit-folium"])
    st.info("Please restart the app after installation completes.")
    maps_dependencies_installed = False

# Check if data is loaded successfully
if neighborhood_data is None or building_age_data is None:
    st.error("Error: Could not load the required data. Please upload the data files or place them in the correct location.")
else:
    # Sidebar
    st.sidebar.header("Property Parameters")
    
    # Get unique neighborhoods
    neighborhoods = ["Altstadt", "Escher Wyss", "Gewerbeschule", "Hochschulen", "Höngg", "Oerlikon", 
                    "Seebach", "Altstetten", "Albisrieden", "Sihlfeld", "Friesenberg", 
                    "Leimbach", "Wollishofen", "Enge", "Wiedikon", "Hard", "Unterstrass", "Oberstrass"]
    selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)
    
    # Room count
    room_count = st.sidebar.slider("Number of Rooms", 1, 6, 3)
    
    # Building age
    building_ages = ["before 1919", "1919-1945", "1946-1970", "1971-1990", "1991-2005", "after 2005"]
    selected_building_age = st.sidebar.selectbox("Building Age", building_ages)
    
    # Travel time preference
    max_travel_time = st.sidebar.slider("Maximum Travel Time (minutes)", 5, 60, 30)
    
    # Predict price
    predicted_price = predict_price(selected_neighborhood, room_count, selected_building_age, max_travel_time)
    
    # Display results
    st.header("Price Prediction")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Estimated Price", f"CHF {predicted_price:,.2f}")
    
    # Display travel times
    if travel_times and selected_neighborhood in travel_times:
        st.header("Travel Times")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Travel Times from Selected Neighborhood")
            for destination, time in travel_times[selected_neighborhood].items():
                st.write(f"{destination}: {time} minutes")
        
        with col2:
            # Create sample travel time chart
            destinations = list(travel_times[selected_neighborhood].keys())
            times = list(travel_times[selected_neighborhood].values())
            
            fig = px.bar(
                x=destinations,
                y=times,
                labels={"x": "Destination", "y": "Travel Time (minutes)"},
                title=f"Travel Times from {selected_neighborhood}"
            )
            st.plotly_chart(fig)
    
    # Display interactive price map
    st.header("Price Heatmap")
    
    # Check if maps functionality is available
    if maps_available and maps_dependencies_installed:
        # Calculate prices for all neighborhoods
        neighborhood_prices = predict_prices_for_all_neighborhoods(
            predict_price, room_count, selected_building_age, max_travel_time
        )
        
        # Display the interactive map
        display_map(neighborhood_prices)
    else:
        st.info("Interactive map visualization would be shown here with real data.")
        st.error("Map functionality is not available. Please make sure maps.py is in the current directory and dependencies are installed.")
        
        # Create a simple bar chart as fallback
        neighborhood_prices = {}
        for neighborhood in neighborhoods:
            neighborhood_prices[neighborhood] = predict_price(neighborhood, room_count, selected_building_age, max_travel_time)
        
        df_prices = pd.DataFrame({
            "Neighborhood": list(neighborhood_prices.keys()),
            "Price (CHF)": list(neighborhood_prices.values())
        })
        
        fig = px.bar(
            df_prices,
            x="Neighborhood",
            y="Price (CHF)",
            title=f"Price Comparison by Neighborhood ({room_count} rooms, {selected_building_age})"
        )
        st.plotly_chart(fig)

# Show setup instructions
with st.expander("App Setup Instructions"):
    st.write("""
    ## How to fix this app:
    
    1. **Data Files**: The app looks for these CSV files in the `data/raw` directory:
       - `bau515od5155.csv` (Property Prices by Neighborhood)
       - `bau515od5156.csv` (Property Prices by Building Age)
       
       You can either:
       - Move them there manually
       - Upload them using the file uploader above
       - Let the app automatically move them if they're in the current directory
    
    2. **Travel Time Data**: The app will automatically generate sample travel time data if not found
    
    3. **Model**: The app will automatically create a simple model if not found
    
    4. **Maps**: For interactive maps, you need:
       - `maps.py` file in the current directory
       - `folium` and `streamlit-folium` packages installed
       
       The app will attempt to install the required packages if they're not available.
    
    All of these files will be generated on first run if not available. For a production app, you would
    replace the sample data generation with actual API calls and model training.
    """)
