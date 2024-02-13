# Train the model for level 2 new data same model
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
from params import device, NEW_SEQ
from model import HybridLSTM, HybridGRU
from dataset import SplitDataset
from labeling import label_mapping, label_removing
from training import training_hybrid
from testing import testing_hybrid

# Define the path
path = 'data/'
md_path = 'new_model_pth/'

# Mixing training and testing together
def train_and_test(model_name = 'Hybrid_LSTM_trans',
    hidden_layers = 64,
    num_layers = 2, # 3
    lr_rate = 5e-3, # 0.005,
    batch_size = 64, 
    fc_layer = 64,
    step_size = 5, gamma = 0.5, l2_lambda = 0.001, min_grp = 50, patience = 2,
    remove_min_grp = True, md_param = ''
):
    
    num_epochs = 200
    
    # Sequence data with sequence length 
    seq_data = np.load(f'{path}/TransData_{NEW_SEQ}.npy', allow_pickle=True) # (534099, 5, 10)
    x_data = np.zeros((seq_data.shape[0], NEW_SEQ, seq_data.shape[2]+1))
    y_data = label_mapping(list(seq_data[:, -1, -2]), list(seq_data[:, -1, -1]))
    
    
    # Data Processing
    for i in range(x_data.shape[0]):
        for seq in range(NEW_SEQ):
            # Static features: Gender, Age, Work
            x_data[i, seq, 0] = seq_data[i, seq*2, 1] # Gender
            x_data[i, seq, 1] = seq_data[i, seq*2, 2] # Age
            x_data[i, seq, 2] = seq_data[i, seq*2, 3] # Work
            
            # Dynamic features: Time Stay, Time Interval for moving
            x_data[i, seq, 3] = datetime.strptime(seq_data[i, seq*2 + 1, 0], '%Y-%m-%d %H:%M:%S').minute - datetime.strptime(seq_data[i, seq*2, 0], '%Y-%m-%d %H:%M:%S').minute
            x_data[i, seq, 4] = datetime.strptime(seq_data[i, (seq+1)*2, 0], '%Y-%m-%d %H:%M:%S').minute - datetime.strptime(seq_data[i, seq*2 + 1, 0], '%Y-%m-%d %H:%M:%S').minute
            x_data[i, seq, 5] = seq_data[i, seq*2 + 1, 4] # Trip Purpose
            x_data[i, seq, 6] = seq_data[i, seq*2 + 1, 5] # Transport Type
            x_data[i, seq, 7] = seq_data[i, seq*2 + 1, 6] # Longitude
            x_data[i, seq, 8] = seq_data[i, seq*2 + 1, 7] # Latitude
            x_data[i, seq, 9] = seq_data[i, seq*2 + 1, 8] # Landuse
            x_data[i, seq, 10] = datetime.strptime(seq_data[i, seq*2 + 1, 0], '%Y-%m-%d %H:%M:%S').hour # Go Hour
            
            
    # Create a StandardScaler to x data
    x_scaler = StandardScaler()
    x_2d = x_data.reshape((x_data.shape[0] * x_data.shape[1], x_data.shape[2]))

    """ Standard Scaler """
    scaled_data_2d = x_scaler.fit_transform(x_2d)
    seq_x_scaled = scaled_data_2d.reshape((x_data.shape[0], x_data.shape[1], x_2d.shape[1]))


    # Find minimum group of y which are less than 10 and remove them
    if remove_min_grp:
        y_minimum_grp = [i for i in set(y_data) if y_data.count(i) < min_grp]
        mask = [False if y in y_minimum_grp else True for y in y_data]
        seq_x_scaled = seq_x_scaled[mask]
        y_data = list(np.array(y_data)[mask])
        y_data = label_removing(y_data)

    # Train Test Valid split with 60:20:20
    train_data, X_sub, train_labels, y_sub = train_test_split(seq_x_scaled, y_data, test_size=0.4, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)
    val_labels = torch.LongTensor(val_labels).to(device)#.float()
    test_labels = torch.LongTensor(test_labels).to(device)#.float()

    # Create a DataLoader for the dataset
    train_seq = train_data[:, :, 3:]
    train_static = train_data[:, 0, :3]
    val_seq = torch.FloatTensor(val_data[:, :, 3:]).to(device)
    val_static = torch.FloatTensor(val_data[:, 0, :3]).to(device)
    test_seq = torch.FloatTensor(test_data[:, :, 3:]).to(device)
    test_static = torch.FloatTensor(test_data[:, 0, :3]).to(device)
    
    # Dataset: sequence_data, static_feature_data, label_data
    train_dataset = SplitDataset(train_seq, train_static, train_labels) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create LSTM model, optimizer, and loss function
    if model_name == 'Hybrid_LSTM_trans':
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=len(set(y_data)), static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
    
    
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
    torch.save(model.state_dict(), os.path.join(md_path, f'{model_name}_{NEW_SEQ}_{md_param}.pth'))

    # Basic testing
    if model_name == 'Hybrid_LSTM_trans':
        model = HybridLSTM(input_size=train_seq.shape[2], hidden_size=hidden_layers, num_layers=num_layers, 
                           output_size=len(set(y_data)), static_feature_size=train_static.shape[1], fc_layer=fc_layer).to(device)
   
    model.load_state_dict(torch.load(os.path.join(md_path, f'{model_name}_{NEW_SEQ}_{md_param}.pth')))
    model.eval()
    return testing_hybrid(model, criterion, test_seq, test_static, test_data, test_labels, x_scaler, len(set(y_data)), md_param)


# Tuning model
if __name__ == '__main__':
    hl_list = [64, 128]
    fc_list = [64]
    lr_list = [0.005, 0.01]
    nl_list = [3] 
    for hl in hl_list:
        for fc in fc_list:
            for lr in lr_list:
                for nl in nl_list:
                    print('----------------------------------')
                    print(f'hl_{hl}_fc_{fc}_lr_{lr}_nl_{nl}')
                    train_and_test(model_name = 'Hybrid_LSTM_trans', 
                                hidden_layers = hl,
                                lr_rate = lr,
                                fc_layer = fc,
                                num_layers = nl,
                                md_param = f'hl_{hl}_fc_{fc}_lr_{lr}_nl_{nl}')