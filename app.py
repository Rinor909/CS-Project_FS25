import streamlit as st
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the app directory to the Python path
app_dir = os.path.join(current_dir, "app")
if os.path.exists(app_dir) and app_dir not in sys.path:
    sys.path.append(app_dir)

# Import the main function from the app module
try:
    from app.app import main
except ImportError as e:
    st.error("Could not import the main function from the 'app' module.")
    st.write("Ensure that your directory structure is correct and that the 'app.py' file exists in the 'app' folder.")
    st.write(f"Detailed error: {str(e)}")
    st.stop()

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.write("Please ensure that all necessary data files are available.")
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.write(f"Detailed error: {str(e)}")
        st.write("Please make sure you've run all the necessary data preparation scripts.")