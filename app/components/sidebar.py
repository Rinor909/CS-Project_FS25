import streamlit as st

def create_sidebar(quartier_mapping):
    """Erstellt die Seitenleiste mit allen Benutzereingaben
    
    Args:
        quartier_mapping: Dictionary mit Mapping zwischen Quartier-Codes und -Namen
        
    Returns:
        Tuple mit: selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport
    """
    
    with st.sidebar:
        # Platz am Anfang hinzufügen für bessere Optik
        st.write("")
        
        # Quartierauswahl - Umkehrung des Mappings für Benutzerfreundlichkeit
        inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}  # Kehrt das Mapping um (Code -> Name zu Name -> Code)
        quartier_options = sorted(inv_quartier_mapping.keys())  # Alphabetisch sortierte Liste aller Quartiere
        
        # Immobilienstandort mit Container-Styling für bessere visuelle Gruppierung
        with st.container(border=True):
            st.subheader("Immobilienstandort")
            # Dropdown zur Quartierauswahl
            selected_quartier = st.selectbox(
                "Der Immobilienstandort geht hier hin:",
                options=quartier_options,
                index=0  # Standardmäßig erstes Quartier ausgewählt
            )
            
            # Quartier-Code für das ausgewählte Quartier abrufen
            quartier_code = inv_quartier_mapping.get(selected_quartier, 0)  # Fallback auf 0, falls Quartier nicht im Mapping
        
        # Platz für visuelle Trennung
        st.write("")
        
        # Immobiliendetails in eigenem Container
        with st.container(border=True):
            # Größenschieberegler - Auswahl der Zimmeranzahl
            st.subheader("Zimmeranzahl")
            selected_zimmer = st.slider(
                "",
                min_value=1,          # Minimum 1 Zimmer
                max_value=6,          # Maximum 6 Zimmer
                value=3,              # Standardwert 3 Zimmer
                step=1,               # In 1er-Schritten
                format="%d Zimmer"    # Angezeigtes Format
            )
            
            # Baujahr mit Dropdown - für Altersauswahl der Immobilie
            st.subheader("Baujahr")
            selected_baujahr = st.selectbox(
                "",
                options=list(range(1900, 2026, 5)),  # Jahre von 1900 bis 2025 in 5er-Schritten
                index=25,                           # Standardwert 2010 (25. Eintrag ab 1900 in 5er-Schritten)
                format_func=lambda x: str(x),       # Formatierung als String
            )
        
        # Transportmittel - in eigenem Container für visuelle Gruppierung
        with st.container(border=True):
            st.subheader("Transportmittel")
            # Horizontale Radiobuttons zur Transportauswahl
            selected_transport = st.radio(
                "",
                options=["öffentlicher Verkehr", "Auto"],
                horizontal=True,
                index=0  # Standard ist öffentlicher Verkehr
            )
        
        # Zuordnung der Auswahlwerte zu API-Keys
        # Dies übersetzt die benutzerfreundlichen Namen in technische Werte für die API
        selected_transport = "transit" if selected_transport == "Public Transit" else "driving"
    
    # Gibt alle Benutzerauswahlen zurück
    return selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport