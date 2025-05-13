import streamlit as st

def create_header():
    """Erstellt den Anwendungs-Header mit Logo und Titel"""
    # Zweispaltiges Layout für den Header
    header_col1, header_col2 = st.columns([1, 3])  # Verhältnis 1:3 für Logo zu Text
    
    with header_col1:
        # Logo-Bereich - Lädt das Logo von einer externen URL
        st.image("https://i.ibb.co/Fb2X2QRB/Logo-Immo-Insight-ZH-w-bg.png", width=300)
    
    with header_col2:
        # Titel und Untertitel der Anwendung
        st.title("Zürcher Immobilien. Datenbasiert. Klar.")
        st.caption("Immobilienpreise in Zürich datengetrieben prognostizieren.")
    
    # Trennlinie hinzufügen - für visuelle Trennung zwischen Header und Inhalt
    st.divider()