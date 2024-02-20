import streamlit as st
import Landingpage  # Assuming this is your landing page script
import NND_Page# Assuming this is another page script
from Book1.Chapter2 import One_input_neuron
from Book1.Chapter2 import Two_input_neuron


# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Function to load a page based on session state
def load_page(page_name):
    if page_name == 'landing':
        Landingpage.run()
    elif page_name == 'nnd':
        NND_Page.run()
    elif page_name == "One_input_neuron":
        One_input_neuron.run()
    elif page_name == "Two_input_neuron":
        Two_input_neuron.run()
        

    # Add more pages as elif blocks here

# Load the current page
load_page(st.session_state.page)
