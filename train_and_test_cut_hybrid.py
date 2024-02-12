# Train the model for 15 * 15 region cut with Train valid test split 60:20:20
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
from itertools import chain
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from model import HybridLSTM
from dataset import SplitDataset
from params import device, SEQ_LENGTH, CUT
from training import training_hybrid
from testing import testing_cut_hybrid
from labeling import region2id
from JP_Code.get_frame import min_lon, min_lat, max_lon, max_lat

path = 'data/'
md_path = 'model_pth/'


# Mixing training and testing together
def train_and_test(model_name = 'LSTM',
    hidden_layers = 64,
    num_layers = 2,
    lr_rate = 0.005,
    batch_size = 64,
    fc_layer = 64,
    step_size = 5,
    gamma = 0.5,
    patience = 5,
    l2_lambda = 0.001,
    min_grp = 50,
    plotting = False, ohe = False, smote = False, remove_min_grp = True,
    grp = ''
):

    LOOKBACK = SEQ_LENGTH - 1
    num_epochs = 200

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

   
    # Filter out the outliers
    y_mask = []
    for i in range(seq_y.shape[0]):
        if seq_y[i,0] >= min_lon and seq_y[i,0] <= max_lon and seq_y[i,1] >= min_lat and seq_y[i,1] <= max_lat:
            y_mask.append(True)
        else:
            y_mask.append(False)
    
    seq_x = seq_x[y_mask]
    seq_y = seq_y[y_mask]
      
    # Create a MinMaxScaler to y data
    scaler = MinMaxScaler()
    seq_y1 = seq_y[:, 0].reshape(-1, 1)
    seq_y2 = seq_y[:, 1].reshape(-1, 1) 
    seq_y1_scale = scaler.fit_transform(seq_y1) #.reshape(seq_y.shape[0], seq_y.shape[1])
    seq_y2_scale = scaler.fit_transform(seq_y2) #.reshape(seq_y.shape[0], seq_y.shape[1])
    y_cluster = [region2id(y1, CUT) * CUT + region2id(y2, CUT) for y1, y2 in zip(seq_y1_scale, seq_y2_scale)]

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


   # Train Test Valid split with 60:20:20
    train_data, X_sub, train_labels, y_sub = train_test_split(seq_x_scaled, y_cluster, test_size=0.4, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)
    val_labels = torch.LongTensor(val_labels).to(device)#.float()
    test_labels = torch.LongTensor(test_labels).to(device)#.float()



    """ Apply SMOTE Oversampling to Training data. """
    if smote:
        print(f"Data size before SMOTE: {len(train_labels)}")
        train_reshaped = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        smote = SMOTE(k_neighbors=4, random_state=42)
        train_resampled, train_labels = smote.fit_resample(train_reshaped, train_labels)
        train_data = train_resampled.reshape(train_resampled.shape[0], train_data.shape[1], train_data.shape[2])
        print(f"Data size after SMOTE: {len(train_labels)}")
        

    # Create a DataLoader for the dataset
    train_seq = np.concatenate((train_data[:, :, 0:1], train_data[:, :, 4:]), axis=2)
    train_static = train_data[:, 0, 1:4] # x1, x2, x3
    val_seq = torch.FloatTensor(np.concatenate((val_data[:, :, 0:1], val_data[:, :, 4:]), axis=2)).to(device)
    val_static = torch.FloatTensor(val_data[:, 0, 1:4]).to(device)
    test_seq = torch.FloatTensor(np.concatenate((test_data[:, :, 0:1], test_data[:, :, 4:]), axis=2)).to(device)
    test_static = torch.FloatTensor(test_data[:, 0, 1:4]).to(device)
    
    # Dataset: sequence_data, static_feature_data, label_data
    train_dataset = SplitDataset(train_seq, train_static, train_labels) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Create LSTM model, optimizer, and loss function
    if model_name == 'Hybrid_LSTM':
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=CUT*CUT, static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
    

    # Define the loss function with weights
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training
    train_losses = []
    pat, best_val = 0, 1e10
    
    for epoch in tqdm(range(num_epochs)):
        tr_loss = training_hybrid(train_loader, model, criterion, optimizer, l2_lambda=l2_lambda)
        train_losses.append(tr_loss)

        scheduler.step()

        # Early stopping Validation
        if (epoch) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(val_seq, val_static)
                val_loss = criterion(outputs, val_labels)
                if val_loss.item() < best_val:
                    pat, best_val = 0, val_loss.item()
                else:
                    pat += 1
                    if pat >= patience: break


    # Store the model
    torch.save(model.state_dict(), os.path.join(md_path, f'{model_name}_model_seq_{SEQ_LENGTH}_cut_{CUT}_0208.pth'))

    # Basic testing
    if model_name == 'Hybrid_LSTM':
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=CUT*CUT, static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
    
    model.load_state_dict(torch.load(os.path.join(md_path, f'{model_name}_model_seq_{SEQ_LENGTH}_cut_{CUT}_0208.pth')))
    model.eval()
    return testing_cut_hybrid(model, criterion, test_seq, test_static, test_labels, plotting)


if __name__ == '__main__':
    train_and_test(model_name = 'Hybrid_LSTM')