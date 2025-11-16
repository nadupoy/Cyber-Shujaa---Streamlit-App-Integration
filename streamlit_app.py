import streamlit as st
import joblib
import pandas as pd
import numpy as np


# 1. Load the saved ML model:

# Use st.cache_resource for models/data that are loaded once:
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

# Set up the app title and configuration:
st.set_page_config(page_title="California Housing Price Predictor", layout="wide")
st.markdown("---")


# 2. Create an intuitive UI for users to input data:
st.sidebar.header("Input House Features")

# Create input widgets using st.sidebar:
MedInc = st.sidebar.slider(
  "Median Income (in $10,000s)",
  min_value=0.5,
  max_value=15.0,
  value=3.0,
  step=0.1
)

HouseAge = st.sidebar.slider(
  "House Age (Median Years)",
  min_value=1.0,
  max_value=55.0,
  value=25.0,
  step=1.0
)

AveRooms = st.sidebar.slider(
  "Average Rooms per Household",
  min_value=1.0,
  max_value=15.0,
  value=5.0,
  step=0.1
)

AveBedrms = st.sidebar.slider(
  "Average Bedrooms per Household",
  min_value=0.5,
  max_value=5.0,
  value=1.0,
  step=0.1
)

Population = st.sidebar.slider(
  "Block Population",
  min_value=10.0,
  max_value=36000.0,
  value=1500.0,
  step=100.0
)

AveOccup = st.sidebar.slider(
  "Average Household Occupancy",
  min_value=0.5,
  max_value=20.0,
  value=3.0,
  step=0.1
)

Latitude = st.sidebar.slider(
  "Latitude",
  min_value=32.0,
  max_value=42.0,
  value=34.0,
  step=0.01
)

Longitude = st.sidebar.slider(
  "Longitude",
  min_value=125.0,
  max_value=114.0,
  value=118.0,
  step=0.01
)

# Compile inputs into a dictionary:
input_data = {
  'MedInc': MedInc,
  'HouseAge': HouseAge,
  'AveRooms': AveRooms,
  'AveBedrms': AveBedrms,
  'Population': Population,
  'AveOccup': AveOccup,
  'Latitude': Latitude,
  'Longitude': Longitude
}

# Convert dictionary to a DataFrame:
input_df = pd.DataFrame([input_data])


# 3. Display predictions and model performance metrics interactively:
st.header("Prediction Results")

if pipeline is not None:
  # Display the input data used for prediction:
  st.subheader("Input Features")
  st.table(input_df)

  # Make prediction:
  prediction = pipeline.predict(input_df)[0]

  # Target variable is Median House Value, scaled in $100,000s:
  predicted_price = prediction * 100000

  # Display the result:
  st.subheader("Predicted Median House Value")

  # Use st.metric for a bold, clear display:
  st.metric(
    label="Predicted Price Estimate",
    value=f"${predicted_price:,2f}"
  )

  st.caption(
    "Note: Price is estimated in USD (2019 values). The model output is scaled by $100,000."
  )

  # 4. (Optional) Display model performance:
  st.markdown("---")
  st.subheader("Model Performance Context")