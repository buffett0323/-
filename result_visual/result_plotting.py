import sys
import torch
import joblib
import torch.nn as nn
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler


sys.path.append('../') 
from model import HybridLSTM, HybridGRU
from dataset import SplitDataset
from labeling import label_mapping, label_removing
from training import training_hybrid
from testing import testing_hybrid
from params import SEQ_LENGTH, device

LOOKBACK = SEQ_LENGTH - 1

# @st.cache(allow_output_mutation=True)
@st.cache_data
def load_model():
    model = HybridGRU(input_size=6, hidden_size=128, num_layers=3, 
                    output_size=227, static_feature_size=3, fc_layer=64).to(device)
   
    model.load_state_dict(torch.load("../new_model_pth/Hybrid_GRU_new_model_4_0429.pth"))  # Load your model weights
    model.eval()
    return model

model = load_model()

# Streamlit interface
st.title("Trip-purpose-based human mobility's next location prediction")
longitude = st.number_input("Enter longitude:", value=120.0)
latitude = st.number_input("Enter latitude:", value=23.5)
submit = st.button("Predict Movement")

# Get x_scaler
scaler_loaded = joblib.load('standard_scaler.pkl')
X_test_scaled = scaler_loaded.transform()

if submit:
    # Assuming model expects [longitude, latitude]
    data = torch.tensor([[longitude, latitude]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(data)
        prediction = prediction.numpy().flatten()
    
    # Show predictions on a map
    fig = px.scatter_geo(lat=[latitude, prediction[1]], lon=[longitude, prediction[0]],
                         locations=[0, 1], locationmode='ISO-3', 
                         color=["Original", "Predicted"], size=[10, 10], 
                         projection="natural earth", title="Prediction on Map")
    st.plotly_chart(fig)
