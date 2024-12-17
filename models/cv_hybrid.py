# Cross validation for y level 2 prediction
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn.model_selection import train_test_split, KFold
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


# Cross validaion for hybrid lstm
def cv_hybrid(model_name = 'Hybrid_LSTM',
    hidden_layers = 128,
    num_layers = 3, 
    lr_rate = 5e-3,
    batch_size = 64, 
    fc_layer = 64,
    step_size = 5, gamma = 0.5, l2_lambda = 0.001, min_grp = 50, patience = 2, n_splits = 5,
    remove_min_grp = True, plotting = False, md_param = ''
):
    
    LOOKBACK = SEQ_LENGTH - 1
    num_epochs = 200
    
    # Sequence data with sequence length
    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}.npy', allow_pickle=True) # (6, 770400, 8)
    seq_data = np.transpose(seq_data, (1, 0, 2))
    
    seq_x = seq_data[:LOOKBACK, :, 1: -2].astype(np.float64) # Temporarily remove 0 (the time)
    seq_y1 = seq_data[LOOKBACK, :, -2].astype(np.float64) # Predict the box
    seq_y2 = seq_data[LOOKBACK, :, -1].astype(np.float64)
    seq_x = np.transpose(seq_x, (1, 0, 2))

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
    y_cluster = label_mapping(list(seq_y1), list(seq_y2))


    # Create a StandardScaler to x data
    x_scaler = StandardScaler()
    seq_x_2d = seq_x.reshape((seq_x.shape[0] * seq_x.shape[1], seq_x.shape[2]))

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


    """ Start Cross Validation """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_tensor = seq_x_scaled 
    y_tensor = np.array(y_cluster)

    # Cross Validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(X_tensor), 1):
        print('----------------------------------')
        print(f"Fold {fold}/{n_splits}")

        # Splitting data into training and validation sets for the current fold
        tmp_data, test_data = X_tensor[train_index], X_tensor[test_index]
        tmp_labels, test_labels = y_tensor[train_index], y_tensor[test_index]
        
        # Val data
        train_data, val_data, train_labels, val_labels = train_test_split(tmp_data, tmp_labels, test_size=0.25, random_state=42)
        val_labels = torch.LongTensor(val_labels).to(device)#.float()
        test_labels = torch.LongTensor(test_labels).to(device)#.float()

        # Processing
        train_seq = train_data[:, :, [0] + list(range(4, train_data.shape[2]))]
        train_static = train_data[:, 0, 1:4]
        val_seq = torch.FloatTensor(val_data[:, :, [0] + list(range(4, val_data.shape[2]))]).to(device)
        val_static = torch.FloatTensor(val_data[:, 0, 1:4]).to(device)
        test_seq = torch.FloatTensor(test_data[:, :, [0] + list(range(4, test_data.shape[2]))]).to(device)
        test_static = torch.FloatTensor(test_data[:, 0, 1:4]).to(device)
    
        # Create datasets and dataloaders
        train_dataset = SplitDataset(train_seq, train_static, train_labels)  
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create Hybrid LSTM model, optimizer, and loss function
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                        output_size=len(set(y_cluster)), static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
         
        # Define the loss function with weights
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)


        # Training loop for the current fold
        pat, best_val = 0, 1e10
        for epoch in tqdm(range(num_epochs)):
            
            training_hybrid(train_loader, model, criterion, optimizer, l2_lambda=l2_lambda)
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
        torch.save(model.state_dict(), os.path.join(md_path, f'{model_name}_model_{SEQ_LENGTH}_fold_{fold}.pth'))

        # Test the model
        model.load_state_dict(torch.load(os.path.join(md_path, f'{model_name}_model_{SEQ_LENGTH}_fold_{fold}.pth')))
        model.eval()
        testing_hybrid(model, criterion, test_seq, test_static, test_labels, len(set(y_cluster)), md_param, plotting)
        

    
# Tuning model
if __name__ == '__main__':
    cv_hybrid()