"""The predicted result is bad."""
from shapely.geometry import Point
from params import device, SEQ_LENGTH
from dataset import RegDataset

import os
import torch
import pandas as pd
import geopandas as gpd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from params import device, SEQ_LENGTH
from model import LSTMRegression
from labeling import label_mapping, label_removing, region2id
from testing import testing_hybrid
from test_visualization import model_comp


path = 'data'
md_path = 'model_pth/'


# Mixing training and testing together
def train_and_test(model_name = 'LSTM',
    hidden_layers = 64,
    num_layers = 2,
    ohe = False,
    grp = ''
):

    LOOKBACK = SEQ_LENGTH - 1

    # Sequence data with sequence length
    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}{grp}.npy', allow_pickle=True) # (6, 770400, 8)
    seq_data = np.transpose(seq_data, (1, 0, 2))

    seq_x = seq_data[:LOOKBACK, :, 1: -2].astype(np.float64) # Temporarily remove 0 (the time)
    seq_y = seq_data[LOOKBACK, :, [6, 7]].astype(np.float64) # Predict the box
    seq_x = np.transpose(seq_x, (1, 0, 2))
    seq_y = np.transpose(seq_y, (1, 0))

    # Time lag
    added_x = np.transpose(seq_data[:, :, 0], (1, 0))
    datetime_array = []
    for row in added_x:
        datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in row]
        diff_list = [int((datetime_row[i+1] - datetime_row[i]).seconds / 60 )for i in range(len(datetime_row)-1)]
        datetime_array.append(diff_list)

    # Concatenate the two arrays along the new axis (axis=2)
    dt_arr = np.expand_dims(np.array(datetime_array), axis=2)
    seq_x = np.concatenate((dt_arr, seq_x), axis=2)

    # Create a StandardScaler to x data
    x_scaler = StandardScaler()
    seq_x_2d = seq_x.reshape((seq_x.shape[0] * seq_x.shape[1], seq_x.shape[2]))


    """ One Hot Encoding """
    if ohe:
        ohe_col = [3, 4, 5, 8]
        for i in ohe_col:
            numbers_series = pd.Series(seq_x_2d[:, i].tolist())
            ohe = pd.get_dummies(numbers_series).to_numpy()
            seq_x_2d = np.concatenate((seq_x_2d, ohe), axis=1)

        # Remove the columns that already execute ohe
        columns_to_keep = [i for i in range(seq_x_2d.shape[1]) if i not in ohe_col]
        seq_x_2d = seq_x_2d[:, columns_to_keep]

    """ Standard Scaler """
    scaled_data_2d = x_scaler.fit_transform(seq_x_2d)
    seq_x_scaled = scaled_data_2d.reshape((seq_x.shape[0], seq_x.shape[1], seq_x_2d.shape[1]))



    # Train Test Valid split
    X_train, X_sub, y_train, y_sub = train_test_split(seq_x_scaled, seq_y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)

    test_data = torch.Tensor(X_test).to(device)
    test_labels = torch.LongTensor(y_test).to(device)#.float()


    # Create LSTM model, optimizer, and loss function
    model = LSTMRegression(input_size=seq_x_scaled.shape[2], hidden_size=hidden_layers, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(os.path.join(md_path, f'{model_name}_model_reg_{SEQ_LENGTH}.pth'), map_location=torch.device('cpu')))
    model.eval()
    
    # Get the test loss
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(test_data)
        test_loss = criterion(outputs, test_labels.squeeze(1))
        print(f'The testing loss is {round(test_loss.item(), 2)}.\n')


    """ Get the accuracy back """
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    
    gdf_actual = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in outputs])
    gdf_predicted = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in test_labels])
    gdf_actual.crs = gdf2.crs
    gdf_predicted.crs = gdf2.crs
    
    
    gdf_actual = gpd.sjoin(gdf_actual, gdf2, how='left', predicate='within')
    gdf_predicted = gpd.sjoin(gdf_predicted, gdf2, how='left', predicate='within')

    matches = sum(gdf_actual['GID_2'] == gdf_predicted['GID_2'])

    for i, j in zip(gdf_actual['GID_2'].tolist(), gdf_predicted['GID_2'].tolist()):
        print(i, j)
    print(f"Number of matching pairs: {matches/gdf_predicted.shape[0]}")


train_and_test()