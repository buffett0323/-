# Train the model for level 2 new data same model
# Best model: Hybrid_LSTM_new_model_4_hl_128_fc_64_lr_0.005_nl_3.pth
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from params import device, SEQ_LENGTH
from model import HybridLSTM, HybridGRU
from dataset import SplitDataset
from labeling import label_mapping, label_removing
from training import training_hybrid
from testing import testing_hybrid

# Define the path
path = 'data/'
md_path = 'new_model_pth/'

# Mixing training and testing together
def train_and_test(model_name = 'Hybrid_LSTM_new',
    hidden_layers = 128,
    num_layers = 3,
    fc_layer = 64,
    min_grp = 50,
    ohe = False, remove_min_grp = True,
    md_param = ''
):
    
    LOOKBACK = SEQ_LENGTH - 1
    
    # Sequence data with sequence length
    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}.npy', allow_pickle=True) # (6, 770400, 8)
    seq_data = np.transpose(seq_data, (1, 0, 2))
    
    seq_x = seq_data[:LOOKBACK, :, 1: -3].astype(np.float64) # Temporarily remove 0 (the time)
    seq_y1 = seq_data[LOOKBACK, :, -2].astype(np.float64) # Predict the box
    seq_y2 = seq_data[LOOKBACK, :, -1].astype(np.float64)
    seq_x = np.transpose(seq_x, (1, 0, 2))

    # Time lag
    added_x = np.transpose(seq_data[:, :, 0], (1, 0))
    datetime_array, hour_array = [], []
    for row in added_x:
        datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in row]
        diff_list = [int((datetime_row[i+1] - datetime_row[i]).seconds / 60 )for i in range(len(datetime_row)-1)]
        hour_list = [int(item.hour) for item in datetime_row[:-1]]
        datetime_array.append(diff_list)
        hour_array.append(hour_list)
        

    # Concatenate the two arrays along the new axis (axis=2)
    dt_arr = np.expand_dims(np.array(datetime_array), axis=2)
    hr_arr = np.expand_dims(np.array(hour_array), axis=2)   
    seq_x = np.concatenate((dt_arr, seq_x), axis=2)
    seq_x = np.concatenate((seq_x, hr_arr), axis=2)
    y_cluster = label_mapping(list(seq_y1), list(seq_y2))


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

    # Find minimum group of y which are less than 10 and remove them
    if remove_min_grp:
        y_minimum_grp = [i for i in set(y_cluster) if y_cluster.count(i) < min_grp]
        mask = [False if y in y_minimum_grp else True for y in y_cluster]
        seq_x_scaled = seq_x_scaled[mask]
        y_cluster = list(np.array(y_cluster)[mask])
        y_cluster = label_removing(y_cluster)


    # Train Test Valid split with 60:20:20
    train_data, X_sub, train_labels, y_sub = train_test_split(seq_x_scaled, y_cluster, test_size=0.4, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)
    val_labels = torch.LongTensor(val_labels).to(device)#.float()
    test_labels = torch.LongTensor(test_labels).to(device)#.float()

    # Create a DataLoader for the dataset
    train_seq = np.concatenate((train_data[:, :, 0:1], train_data[:, :, 4:]), axis=2)
    train_static = train_data[:, 0, 1:4]
    test_seq = torch.FloatTensor(np.concatenate((test_data[:, :, 0:1], test_data[:, :, 4:]), axis=2)).to(device)
    test_static = torch.FloatTensor(test_data[:, 0, 1:4]).to(device)

    # Define the loss function with weights
    criterion = nn.CrossEntropyLoss()

    # Basic testing
    if model_name == 'Hybrid_LSTM_new':
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=len(set(y_cluster)), static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
    elif model_name == 'Hybrid_GRU_new':
        model = HybridGRU(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=len(set(y_cluster)), static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
   
    model.load_state_dict(torch.load(os.path.join(md_path, f'{model_name}_model_{SEQ_LENGTH}_{md_param}.pth')))
    model.eval()
    return testing_hybrid(model, criterion, test_seq, test_static, test_data, test_labels, x_scaler, len(set(y_cluster)), md_param)


# Tuning model
if __name__ == '__main__':
    hl = 128
    nl = 3
    fc = 64
    lr = 0.005
    train_and_test(model_name = 'Hybrid_GRU_new', 
                    hidden_layers = hl,
                    fc_layer = fc,
                    num_layers = nl,
                    md_param = 'final')


