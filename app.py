import streamlit as st
import os
import sys

# Add app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "app")
sys.path.append(app_dir)

# Import the main function from app/app.py
from app.app import main

if __name__ == "__main__":
    main()