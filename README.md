# 🏡 Zürcher Immobilienpreis-Prognose

Eine Webanwendung, die Immobilienpreisschätzungen in Zürich basierend auf Lage, Zimmeranzahl und Baujahr anzeigt.

## Was dieses Projekt leistet

Diese App hilft Ihnen:
- Geschätzte Immobilienpreise in verschiedenen Zürcher Quartieren zu sehen
- Preisunterschiede in ganz Zürich auf einer Karte zu betrachten
- Zu überprüfen, wie Reisezeiten die Preise beeinflussen
- Preise zwischen verschiedenen Quartieren zu vergleichen

## Einrichtungsanleitung

1. Klonen Sie dieses Repository
```
git clone https://github.com/yourusername/zurich-real-estate.git
cd zurich-real-estate
```

2. Installieren Sie die erforderlichen Pakete
```
pip install -r requirements.txt
```

3. Fügen Sie Ihren Google Maps API-Schlüssel hinzu
   - Erstellen Sie eine `.streamlit/secrets.toml` Datei
   - Fügen Sie diese Zeile mit Ihrem API-Schlüssel hinzu: `GOOGLE_MAPS_API_KEY = "your-api-key-here"`

## Vorbereiten der Daten

Führen Sie diese Skripte der Reihe nach aus:

```
# 1. Immobiliendaten verarbeiten
python scripts/data_preparation.py

# 2. Reisezeit-Daten generieren
python scripts/generate_travel_times.py

# 3. Vorhersagemodell trainieren
python scripts/model_training.py
```

## Ausführen der App

```
streamlit run app.py
```

Öffnen Sie Ihren Browser und gehen Sie zu `http://localhost:8501`

## Projektdateien

- `app.py`: Haupt-Anwendungsdatei
- `app/`: Verzeichnis mit App-Komponenten
- `data/`: Enthält Roh- und verarbeitete Daten
- `models/`: Speichert das trainierte ML-Modell
- `scripts/`: Enthält Datenverarbeitungsskripte
- `requirements.txt`: Listet alle erforderlichen Pakete auf

## Funktionen

- **Preisvorhersage**: Schätzt Immobilienpreise mit einem maschinellen Lernmodell
- **Interaktive Karten**: Zeigt die Preisverteilung in Zürich
- **Reisezeit-Analyse**: Zeigt, wie Reisezeiten die Immobilienwerte beeinflussen
- **Preisvergleich**: Vergleicht Preise zwischen verschiedenen Quartieren

## Datenquellen

- Immobilienpreise nach Quartier (2009-2024)
- Immobilienpreise nach Baujahr (2009-2024)
- Reisezeiten zu wichtigen Standorten (berechnet mit der Google Maps API)

## Verwendete Tools

- **Python**: Programmiersprache
- **Streamlit**: Weboberfläche
- **Pandas**: Datenverarbeitung
- **Scikit-learn**: Maschinelles Lernen für Vorhersagen
- **Plotly**: Interaktive Diagramme und Karten
- **Google Maps API**: Reisezeit-Daten

## Hinweis

Dieses Projekt wurde für einen Informatikkurs erstellt. Die Vorhersagen sind nur Schätzungen und sollten nicht für tatsächliche Immobilienentscheidungen verwendet werden.
