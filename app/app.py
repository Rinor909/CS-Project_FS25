import streamlit as st
import sys
import os

# Füge das App-Verzeichnis zum Pfad für Importe hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiere Komponenten
from components.header import create_header
from components.sidebar import create_sidebar
from components.tabs import create_tabs
from utils.styling import apply_chart_styling

# Importiere Datendienstprogramme
from utils import (
    load_processed_data, load_model, load_quartier_mapping,
    preprocess_input, predict_price, get_travel_times_for_quartier,
    get_quartier_statistics, get_price_history, get_zurich_coordinates,
    get_quartier_coordinates
)

# Hauptlogik der Anwendung 
def main():
    # Seitenkonfiguration - sauberes und breites Layout
    st.set_page_config(
        page_title="ImmoInsight ZH",           # Titel im Browser-Tab
        page_icon="🦁",                       # Symbol im Browser-Tab (Löwe-Emoji)
        layout="wide",                        # Breites Layout für bessere Visualisierung
        initial_sidebar_state="expanded"      # Seitenleiste standardmäßig ausgeklappt
    )

    # Funktion zum Laden von Daten und Modell mit Caching
    # Caching verbessert die Leistung, indem es verhindert, dass Daten bei jeder Interaktion neu geladen werden
    @st.cache_resource
    def load_data_and_model():
        """Lädt alle Daten und Modelle (mit Caching für bessere Leistung)"""
        # Lädt Basis-Datensätze
        df_quartier, df_baualter, df_travel_times = load_processed_data()
        # Lädt Machine Learning Modell
        model = load_model()
        # Lädt Mapping zwischen Quartier-Codes und -Namen
        quartier_mapping = load_quartier_mapping()
        # Lädt Koordinaten für die Kartendarstellung
        quartier_coords = get_quartier_coordinates()
        
        # Gibt alle geladenen Ressourcen zurück
        return df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords

    # Lädt Daten und Modell - dies wird dank Caching nur einmal ausgeführt
    df_quartier, df_baualter, df_travel_times, model, quartier_mapping, quartier_coords = load_data_and_model()
    
    # Erstellt inverse Mapping (Quartier-Name → Quartier-Code)
    inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
    
    # Überprüft, ob Daten verfügbar sind
    # Zeigt Anweisungen, falls die Basisdaten fehlen
    if df_quartier.empty or 'Quartier' not in df_quartier.columns:
        st.warning("Erforderliche Daten nicht gefunden. Bitte führen Sie zuerst die Datenvorbereitungsskripte aus.")
        st.info("Ausführen: python scripts/data_preparation.py")
        st.info("Ausführen: python scripts/generate_travel_times.py")
        st.info("Ausführen: python scripts/model_training.py")
        return
    
    # Erstellt den Header mit Logo und Titel
    create_header()
    
    # Erstellt die Seitenleiste und holt Benutzerauswahlen
    selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport = create_sidebar(quartier_mapping)
    
    # Reisezeiten für das ausgewählte Quartier abrufen
    # Dies wird für die Preisvorhersage und Reisezeitvisualisierung verwendet
    travel_times = get_travel_times_for_quartier(
        selected_quartier, 
        df_travel_times, 
        transportmittel=selected_transport
    )
    
    # Eingaben für das Modell vorbereiten und Preis vorhersagen
    # Diese Schritte konvertieren die Benutzerauswahl in ein Format, das das ML-Modell verarbeiten kann
    input_data = preprocess_input(
        quartier_code,        # Numerischer Code für das ausgewählte Quartier
        selected_zimmer,      # Ausgewählte Zimmeranzahl
        selected_baujahr,     # Ausgewähltes Baujahr
        travel_times          # Reisezeitdaten für verschiedene Ziele
    )
    # Preis basierend auf den Eingabedaten vorhersagen
    predicted_price = predict_price(model, input_data)
    
    # Erstellt den Hauptcontainer für den Inhalt
    with st.container(border=True):
        # Immobilienbewertungsabschnitt - Hauptergebnis der Anwendung
        st.subheader("Immobilienbewertung")
        
        # Preisanzeige in einem Container
        # Zeigt den vorhergesagten Preis prominent an
        price_container = st.container(border=False)
        price_container.metric(
            label="Geschätzer Immobilienwert",
            value=f"{predicted_price:,.0f} CHF" if predicted_price else "N/A",  # Formatiert Preis mit Tausendertrennzeichen
            delta=f"{round((predicted_price / 1000000 - 1) * 100, 1):+.1f}%" if predicted_price else None,  # Prozentuale Abweichung von 1 Mio.
            delta_color="inverse"  # Rote Farbe bei positiver Abweichung (teurer)
        )
        
        # Erstellt alle Tab-Inhalte
        # Übergibt alle notwendigen Daten und Funktionen an die Tab-Komponente
        create_tabs(
            df_quartier,                      # Quartier-Datensatz
            df_travel_times,                  # Reisezeit-Datensatz
            quartier_coords,                  # Quartier-Koordinaten
            selected_quartier,                # Ausgewähltes Quartier
            selected_zimmer,                  # Ausgewählte Zimmeranzahl
            selected_baujahr,                 # Ausgewähltes Baujahr
            predicted_price,                  # Vorhergesagter Preis
            travel_times,                     # Reisezeiten
            quartier_options=sorted(list(inv_quartier_mapping.keys())),  # Alle verfügbaren Quartiere (mit list() konvertiert)
            apply_chart_styling=apply_chart_styling  # Styling-Funktion
        )
    
    # Fußzeile mit Quellenangaben und Entwicklungskontext
    st.caption(
        "Entwickelt im Rahmen des CS-Kurses an der HSG | Datenquellen: "
        "[Immobilienpreise nach Quartier](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-2) | "
        "[Immobilienpreise nach Baualter](https://opendata.swiss/en/dataset/verkaufspreise-median-pro-wohnung-und-pro-quadratmeter-wohnungsflache-im-stockwerkeigentum-2009-3)"
    )

# Ausführungsprüfung - Code wird nur ausgeführt, wenn die Datei direkt ausgeführt wird
if __name__ == "__main__":
    main()