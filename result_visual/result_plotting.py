import os
import sys
import geopandas as gpd
import torch
import folium
import joblib
import datetime
import torch.nn as nn
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon

sys.path.append('../') 
from model import HybridLSTM, HybridGRU
from dataset import SplitDataset
from labeling import label_mapping, label_removing
from training import training_hybrid
from testing import testing_hybrid
from params import SEQ_LENGTH, device, occupation, trip_purpose, transport_type

LOOKBACK = SEQ_LENGTH - 1

# Match to the y mapping
y_label = []
gdf = gpd.read_file('../../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
with open('../data/y_mapping.txt', 'r') as f:
    for line in f:
        gid_num = int(line.split(':')[-1].split('.')[0])
        gid_num1, gid_num2 = int(gid_num//100), gid_num%100
        y_label.append(f'JPN.{gid_num1}.{gid_num2}_1')



def get_key_by_value(my_dict, search_value):
    for key, value in my_dict.items():
        if value == search_value:
            return key
    return None 


# @st.cache_data
def load_model():
    model = HybridGRU(input_size=6, hidden_size=128, num_layers=3, 
                    output_size=227, static_feature_size=3, fc_layer=64).to(device)
   
    model.load_state_dict(torch.load("../new_model_pth/Hybrid_GRU_new_model_4_0429.pth"))  # Load your model weights
    model.eval()
    return model


# Function to create a Folium map
def create_map(location, initial_zoom, lons, lats):
    # Initial map
    m = folium.Map(location=location, zoom_start=initial_zoom, 
                   control_scale=True)#, tiles='CartoDB positron')
    
    # Add markers
    color_list = ['red', 'lightred', 'orange']
    for i in range(LOOKBACK):
        folium.Marker(
            location=[lats[i], lons[i]],
            popup=f"Sequence_{i+1}",
            icon=folium.Icon(color=color_list[i], icon='info-sign')
        ).add_to(m)
    return m


# Function to add a MultiPolygon to a Folium map
def add_multi_polygon(map_obj, multi_polygon, color):
    for polygon in multi_polygon:
        if isinstance(polygon, MultiPolygon):
            for pg in polygon.geoms:
                if pg.is_empty:
                    continue
                x, y = pg.exterior.xy
                loc = [[i, j] for i, j in zip(list(y), list(x))] # y, x

                folium.Polygon(
                    locations=loc,
                    color='darkgreen',  # Border color
                    fill_color=color,
                    fill_opacity=0.5,
                    weight=2,
                ).add_to(map_obj)

        elif isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            loc = [[i, j] for i, j in zip(list(y), list(x))] # y, x

            folium.Polygon(
                locations=loc,
                color='darkgreen',  # Border color
                fill_color=color,
                fill_opacity=0.5,
                weight=2,
            ).add_to(map_obj)


""" Main """
model = load_model()

# Streamlit interface
st.title("Trip-purpose-based human mobility's next location prediction")

# Static features
gender_dict = dict({1: "Male", 2: "Female", 9: "Unknown"})
gender = get_key_by_value(gender_dict, st.selectbox("Gender:", gender_dict.values()))
age_grp = st.number_input("Age:", value=18) // 5
work = get_key_by_value(occupation, st.selectbox("Work:", occupation.values()))

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

time_last = st.time_input("Choose a time to predict:", datetime.time(10, 0))

# Preprocessing of the data
data = np.zeros((LOOKBACK, 9))
data[0, 0] = (datetime.datetime.combine(datetime.date.today(), time2) - datetime.datetime.combine(datetime.date.today(), time1)).total_seconds()/60
data[1, 0] = (datetime.datetime.combine(datetime.date.today(), time3) - datetime.datetime.combine(datetime.date.today(), time2)).total_seconds()/60
data[2, 0] = (datetime.datetime.combine(datetime.date.today(), time_last) - datetime.datetime.combine(datetime.date.today(), time3)).total_seconds()/60


for i in range(LOOKBACK):
    data[i, 1], data[i, 2], data[i, 3] = gender, age_grp, work

data[0, 4], data[1, 4], data[2, 4] = trip1, trip2, trip3
data[0, 5], data[1, 5], data[2, 5] = trans1, trans2, trans3
data[0, 6], data[1, 6], data[2, 6] = lon1, lon2, lon3
data[0, 7], data[1, 7], data[2, 7] = lat1, lat2, lat3 
data[0, 8], data[1, 8], data[2, 8] = time1.hour, time2.hour, time3.hour
lons = [lon1, lon2, lon3]
lats = [lat1, lat2, lat3]

# Get x_scaler
scaler_loaded = joblib.load('standard_scaler.pkl')
data_scaled = scaler_loaded.transform(data)

# Initialize session state
if 'map_displayed' not in st.session_state:
    st.session_state['map_displayed'] = None


# Submission
if st.button("Predict Movement"):
    sequences = torch.FloatTensor(np.concatenate((data_scaled[:, 0:1], data_scaled[:, 4:]), axis=1).reshape((1, LOOKBACK, 6)))
    static_features = torch.FloatTensor(np.array(data_scaled[0, 1:4]).reshape(1, 3))
    with torch.no_grad():
        outputs = model(sequences, static_features)
        pred = int(torch.argmax(outputs, dim=1)[0])
        pred_gadm = gdf[gdf['GID_2'] == y_label[pred]].geometry


        # Streamlit app
        st.title("Prediction on map")
        location = [35.682839, 139.759455]  # Approximate location of the Imperial Palace in Tokyo
        zoom_start = 9

        # Add the MultiPolygon to the map with light green color
        m = create_map(location, zoom_start, lons, lats)
        add_multi_polygon(m, pred_gadm, 'lightred')
        
        # Store map in session state
        st.session_state['map_displayed'] = m


# Display the map if it is stored in session state
if st.session_state['map_displayed'] is not None:
    st_data = st_folium(st.session_state['map_displayed'], width=725, height=500)
    st.write("Map displayed above shows the predicted region.")
