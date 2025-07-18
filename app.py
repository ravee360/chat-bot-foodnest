import streamlit as st
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Run the Streamlit app directly
st.set_page_config(page_title="DocBot - Document Research & Theme ID", layout="wide")

# Import and run the UI code
import frontend.ui 