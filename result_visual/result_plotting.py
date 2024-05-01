import sys
import torch
import joblib
import datetime
import torch.nn as nn
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler


sys.path.append('../') 
from model import HybridLSTM, HybridGRU
from dataset import SplitDataset
from labeling import label_mapping, label_removing
from training import training_hybrid
from testing import testing_hybrid
from params import SEQ_LENGTH, device, occupation, trip_purpose, transport_type

LOOKBACK = SEQ_LENGTH - 1


def get_key_by_value(my_dict, search_value):
    for key, value in my_dict.items():
        if value == search_value:
            return key
    return None 


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

# Static features
age_grp = st.number_input("Age:", value=18) // 5
gender = st.number_input("Gender:", value=1)
work = st.selectbox("Work:", occupation.values())

# Dynamic features
lon1 = st.number_input("Longitude at Seq1:", value=139.605314)
lat1 = st.number_input("Latitude at Seq1:", value=35.427077)
time1 = st.time_input("Choose a time at Seq1:", datetime.time(8, 50))
trip1 = get_key_by_value(trip_purpose, st.selectbox("Trip at Seq1:", trip_purpose.values()))
trans1 = get_key_by_value(transport_type, st.selectbox("Transport at Seq1:", transport_type.values()))

lon2 = st.number_input("Longitude at Seq2:", value=139.604068)
lat2 = st.number_input("Latitude at Seq2:", value=35.426279)
time2 = st.time_input("Choose a time at Seq2:", datetime.time(9, 5))
trip2 = get_key_by_value(trip_purpose, st.selectbox("Trip at Seq2:", trip_purpose.values()))
trans2 = get_key_by_value(transport_type, st.selectbox("Transport at Seq2:", transport_type.values()))

lon3 = st.number_input("Longitude at Seq3:", value=139.635925)
lat3 = st.number_input("Latitude at Seq3:", value=35.445800)
time3 = st.time_input("Choose a time at Seq3:", datetime.time(9, 25))
trip3 = get_key_by_value(trip_purpose, st.selectbox("Trip at Seq3:", trip_purpose.values()))
trans3 = get_key_by_value(transport_type, st.selectbox("Transport at Seq3:", transport_type.values()))


submit = st.button("Predict Movement")

# Get x_scaler
scaler_loaded = joblib.load('standard_scaler.pkl')
# X_test_scaled = scaler_loaded.transform()

if submit:
    # Assuming model expects [longitude, latitude]
    sequences = torch.FloatTensor([gender, age_grp, work])
    static_features = torch.FloatTensor([])
    with torch.no_grad():
        prediction = model(sequences, static_features)
        prediction = prediction.numpy().flatten()
    
    # Show predictions on a map
    fig = px.scatter_geo(lat=[lat1, prediction[1]], lon=[lon1, prediction[0]],
                         locations=[0, 1], locationmode='ISO-3', 
                         color=["Original", "Predicted"], size=[10, 10], 
                         projection="natural earth", title="Prediction on Map")
    st.plotly_chart(fig)
