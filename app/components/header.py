import streamlit as st

def create_header():
    """Create the application header with logo and title"""
    # Two-column layout for header
    header_col1, header_col2 = st.columns([1, 3])
    
    with header_col1:
        # Logo area
        st.image("https://i.ibb.co/Fb2X2QRB/Logo-Immo-Insight-ZH-w-bg.png", width=300)
    
    with header_col2:
        # Title and subtitle
        st.title("Zürcher Immobilien. Datenbasiert. Klar.")
        st.caption("Immobilienpreise in Zürich datengetrieben prognostizieren.")
    
    # Add a separator
    st.divider()