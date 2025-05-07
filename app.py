import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add app directory to Python path
app_dir = os.path.join(current_dir, "app")
sys.path.append(app_dir)

# For direct app import
try:
    from app.app import main
except ImportError:
    st.error("Could not import main function from app module. Check your directory structure.")
    st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure you've run all the data preparation scripts first.")