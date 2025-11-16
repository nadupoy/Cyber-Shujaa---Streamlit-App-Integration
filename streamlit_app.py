import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model only once and cache it
# Use st.cache_resource for models/data that are loaded once
@st.cache_resource
def load_model(path='california_housing_knr_pipeline.joblib'):
  # Load the trained pipeline:
  try:
    pipeline = joblib.load(path)
    return pipeline
  except FileNotFoundError:
    st.error(f"Error: Model file {path} not found. Please check your file path.")
    return None

pipeline = load_model()