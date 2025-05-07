import streamlit as st
import sys
import os

# Pfade zum Projektverzeichnis hinzufügen
sys.path.append(os.path.abspath("app"))

# App starten
from app.app import main

if __name__ == "__main__":
    main()