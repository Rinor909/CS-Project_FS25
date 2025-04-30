"""
Datenbereinigungsskript für das Immobilienpreisvorhersage-Projekt Zürich

Dieses Skript:
1. Lädt die Datensätze
2. Bereinigt und vereinfacht die Daten
3. Erstellt einen zusammengeführten Datensatz
4. Fügt echte Reisezeiten aus der Google Maps API hinzu
5. Erstellt grundlegende Visualisierungen
"""

import pandas as pd  # Bibliothek für Datenanalyse und -manipulation
import matplotlib.pyplot as plt  # Bibliothek für Visualisierungen
import json  # Bibliothek zum Lesen/Schreiben von JSON-Dateien
import os  # Bibliothek für Dateisystemoperationen
