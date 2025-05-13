import streamlit as st
import sys
import os

# Add the app directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.header import create_header
from components.sidebar import create_sidebar
from components.tabs import create_tabs
from utils.styling import apply_chart_styling

# Import data utilities
from utils import (
    load_processed_data, load_model, load_quartier_mapping,
    preprocess_input, predict_price, get_travel_times_for_quartier,
    get_quartier_statistics, get_price_history, get_zurich_coordinates,
    get_quartier_coordinates
)

# Main app logic 
def main():
    # Page configuration
    st.set_page_config(
        page_title="ImmoInsight ZH",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Function to load data and model with caching
    @st.cache_resource
    def load_data_and_model():
        """Loads all data and models (with caching for performance)"""
        df_quartier, df_baualter, df_travel_times = load_processed_data()
        model = load_model()
        quartier_mapping = load_quartier_mapping()
        quartier_coords = get_quartier_coordinates()
        
        return df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords

    # Load data and model
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Check if data is available
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Required data not found. Please run the data preparation scripts first.")
        st.info("Run: python scripts/data_preparation.py")
        st.info("Run: python scripts/generate_travel_times.py")
        st.info("Run: python scripts/model_training.py")
        return
    
    # Create header with logo and title
    create_header()
    
    # Create sidebar and get user selections
    selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport = create_sidebar(quartier_mapping)
    
    # Get travel times for the selected neighborhood
    travel_times = get_travel_times_for_quartier(
        selected_quartier, 
        df_travel_times, 
        transportmittel=selected_transport
    )
    
    # Prepare inputs for the model and predict price
    input_data = preprocess_input(
        quartier_code, 
        selected_zimmer, 
        selected_baujahr, 
        travel_times
    )
    predicted_price = predict_price(model, input_data)
    
    # Create tabs and content
    with st.container(border=True):
        # Property valuation section
        st.subheader("Immobilienbewertung")
        
        # Price display
        price_container = st.container(border=False)
        price_container.metric(
            label="Geschätzer Immobilienwert",
            value=f"{predicted_price:,.0f} CHF" if predicted_price else "N/A",
            delta=f"{round((predicted_price / 1000000 - 1) * 100, 1):+.1f}%" if predicted_price else None,
            delta_color="inverse"
        )
        
        # Create all tab content
        create_tabs(
            df_quartier, 
            df_travel_times, 
            quartier_coords, 
            selected_quartier, 
            selected_zimmer, 
            selected_baujahr,
            predicted_price, 
            travel_times, 
            quartier_options=sorted(inv_quartier_mapping.keys()), 
            apply_chart_styling=apply_chart_styling
        )
    
    # Footer
    st.caption(
        "Entwickelt im Rahmen des CS-Kurses an der HSG | Datenquellen: "
        "[Immobilienpreise nach Quartier](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-2) | "
        "[Immobilienpreise nach Baualter](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-3)"
    )

if __name__ == "__main__":
    main()