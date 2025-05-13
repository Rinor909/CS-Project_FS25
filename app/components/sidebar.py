import streamlit as st

def create_sidebar(quartier_mapping):
    """Create sidebar with all user inputs"""
    # Return values: selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport
    
    with st.sidebar:
        # Add some space at the top
        st.write("")
        
        # Neighborhood selection
        inv_quartier_mapping = {v: k for k, v in quartier_mapping.items()}
        quartier_options = sorted(inv_quartier_mapping.keys())
        
        # Property location with container styling
        with st.container(border=True):
            st.subheader("Immobilienstandort")
            selected_quartier = st.selectbox(
                "Der Immobilienstandort geht hier hin:",
                options=quartier_options,
                index=0
            )
            
            # Get quartier code
            quartier_code = inv_quartier_mapping.get(selected_quartier, 0)
        
        # Add some space
        st.write("")
        
        # Property details
        with st.container(border=True):
            # Size slider
            st.subheader("Zimmeranzahl")
            selected_zimmer = st.slider(
                "",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                format="%d Zimmer"
            )
            
            # Construction year with dropdown
            st.subheader("Baujahr")
            selected_baujahr = st.selectbox(
                "",
                options=list(range(1900, 2026, 5)),
                index=25,  # Default Baujahr to 2010
                format_func=lambda x: str(x),
            )
        
        # Transportation mode - Put this in its own container
        with st.container(border=True):
            st.subheader("Transportmittel")
            selected_transport = st.radio(
                "",
                options=["öffentlicher Verkehr", "Auto"],
                horizontal=True,
                index=0
            )
        
        # Map selection values back to original keys
        selected_transport = "transit" if selected_transport == "Public Transit" else "driving"
    
    return selected_quartier, quartier_code, selected_zimmer, selected_baujahr, selected_transport