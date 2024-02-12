# Train the model for level 2
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
from imblearn.under_sampling import RandomUnderSampler
from test_visualization import visualize_cut_result, visualize_admin_result
from params import SEQ_LENGTH, occupation, trip_purpose, transport_type


path = 'data/'
device = ''
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")
    
    

class LSTM_Cluster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.2):
        super(LSTM_Cluster, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = drop_prob).to(device)
        self.bn = nn.BatchNorm1d(hidden_size).to(device)  # Batch normalization layer
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        out = self.fc(self.relu(out[:, -1, :]))
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_prob=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = drop_prob).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(self.relu(out[:, -1, :]))  # Get the output from the last time step
        return out


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.data[idx])).to(device), torch.LongTensor(np.array([self.labels[idx]])).to(device)


# Testing the model result
def testing(model, criterion, test_data, test_labels, clust_cnt, x_scaler, plotting=False):
    
    # Filter the test data
    test_2d = test_data.reshape((test_data.shape[0] * test_data.shape[1], test_data.shape[2]))
    test_2d_orig = x_scaler.inverse_transform(test_2d)
    test_data_orig = test_2d_orig.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    
    last_occ, last_tp, last_trans = test_data_orig[:, -1, 3].tolist(), test_data_orig[:, -1, 4].tolist(), test_data_orig[:, -1, 5].tolist()
    last_occ, last_tp, last_trans = [int(round(i, 0)) for i in last_occ], [int(round(i, 0)) for i in last_tp], [int(round(i, 0)) for i in last_trans]
    acc_occ, acc_tp, acc_trans = [], [], []
    test_data = test_data[:, :-1, :]
    occ_value, tp_value = [v for _, v in occupation.items()], [v for _, v in trip_purpose.items()]
    
    for occ, _ in occupation.items():

        mask = [True if i == occ else False for i in last_occ]
        test_data1 = test_data[mask]
        test_labels1 = test_labels[mask]
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_data1)
            pred = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(test_labels1.cpu(), pred.cpu())
            acc_occ.append(accuracy)
    
    plt.figure(figsize=(10, 8))
    plt.plot(acc_occ)
    plt.xticks(range(len(acc_occ)), occ_value, rotation=90, fontsize=5, ha='right') 
    plt.ylabel('Accuracy')
    plt.title('The accuracy of different occupation')
    plt.tight_layout()
    for i in range(len(acc_occ)):
        plt.text(i, acc_occ[i], str(round(acc_occ[i], 3)), ha='center', va='bottom', fontsize=8)
    plt.savefig(f'visualization/acc_occ.jpg')
    plt.show()
    
    for tp, _ in trip_purpose.items():
        
        mask = [True if i == tp else False for i in last_tp]
        test_data1 = test_data[mask]
        test_labels1 = test_labels[mask]
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_data1)
            pred = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(test_labels1.cpu(), pred.cpu())
            acc_tp.append(accuracy)
    
    plt.figure(figsize=(10, 8))
    plt.plot(acc_tp)
    plt.xticks(range(len(acc_tp)), tp_value, rotation=90, fontsize=5, ha='right')
    plt.tight_layout()
    plt.ylabel('Accuracy')
    plt.title('The accuracy of different trip purpose')
    for i in range(len(acc_tp)):
        plt.text(i, acc_tp[i], str(round(acc_tp[i], 3)), ha='center', va='bottom', fontsize=8)
    plt.savefig(f'visualization/acc_tp.jpg')
    plt.show()
    
    
    trans_value = []
    for tr, trans_type in transport_type.items():

        mask = [True if i == tr else False for i in last_trans]
        if mask.count(True) == 0:
            continue
        test_data1 = test_data[mask]
        test_labels1 = test_labels[mask]
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_data1)
            pred = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(test_labels1.cpu(), pred.cpu())
            acc_trans.append(accuracy)
            trans_value.append(trans_type)

    plt.figure(figsize=(10, 8))
    plt.plot(acc_trans)
    plt.xticks(range(len(acc_trans)), trans_value, rotation=90, fontsize=5, ha='right') 
    plt.tight_layout()
    plt.ylabel('Accuracy')
    plt.title('The accuracy of different transport type')
    for i in range(len(acc_trans)):
        plt.text(i, acc_trans[i], str(round(acc_trans[i], 3)), ha='center', va='bottom', fontsize=8)
    plt.savefig(f'visualization/acc_trans.jpg')
    plt.show()
    
    return


def label_mapping(label1, label2):
    old_label = [100*y1 + y2 for y1, y2 in zip(label1, label2)]
    new_label = []
    exist_labels = list(set(old_label))
    mapping = {exist_labels[i]: i for i in range(len(exist_labels))}

    # Store the mapping results
    with open(os.path.join(path, 'y_mapping.txt'), 'w') as f:
        for key, value in mapping.items():
            f.write(f"{value}:{key}\n")

    for i in range(len(old_label)):
        new_label.append(mapping[old_label[i]])
    if len(old_label) == len(new_label):
        return new_label
    return []


def label_removing(old_label):
    old_label, new_label = [int(i) for i in old_label], []
    exist_labels = list(set(old_label))
    mapping = {exist_labels[i]: i for i in range(len(exist_labels))} # 0:0, 3:3, 5:4, 7:5, ..., 233:213

    y_init, y_label = [], []
    with open(os.path.join(path, 'y_mapping.txt'), 'r') as f:
        for line in f:
            y_init.append(int(line.split(':')[0])) # 0-236
            y_label.append(float(line.split(':')[-1].split('\n')[0])) # GID
    
    new_mapping = {mapping[i]: j for i, j in zip(y_init, y_label) if i in exist_labels}
    with open(os.path.join(path, 'y_mapping.txt'), 'w') as f:
        for key, value in new_mapping.items():
            f.write(f"{key}:{value}\n")

    for i in range(len(old_label)):
        new_label.append(mapping[old_label[i]])
    if len(old_label) == len(new_label):
        return new_label
    return []


# Mixing training and testing together
def train_and_test(model_name = 'LSTM',
    hidden_layers = 64,
    num_layers = 2,
    plotting = False, ohe = True, smote = True, under_sampling = False, remove_min_grp = False,
    min_grp = 50,
    grp = ''
):

    LOOKBACK = SEQ_LENGTH - 1

    # Sequence data with sequence length
    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}{grp}.npy', allow_pickle=True) # (6, 770400, 8)
    # mask = [True if i < 10000  else False for i in range(seq_data.shape[0])] # mask
    # seq_data = seq_data[mask]
    seq_data = np.transpose(seq_data, (1, 0, 2))

    seq_x = seq_data[:, :, 1: -2].astype(np.float64) # Temporarily remove 0 (the time)
    # seq_x = seq_data[:LOOKBACK, :, 1: -2].astype(np.float64) # Temporarily remove 0 (the time)
    seq_y1 = seq_data[LOOKBACK, :, -2].astype(np.float64) # Predict the box
    seq_y2 = seq_data[LOOKBACK, :, -1].astype(np.float64)
    seq_x = np.transpose(seq_x, (1, 0, 2))

    # Time lag
    added_x = np.transpose(seq_data[:, :, 0], (1, 0))
    datetime_array = []
    for row in added_x:
        datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in row]
        diff_list = [int((datetime_row[i+1] - datetime_row[i]).seconds / 60 )for i in range(len(datetime_row)-1)]
        diff_list.append(0) # ADDED
        datetime_array.append(diff_list)

    # Concatenate the two arrays along the new axis (axis=2)
    dt_arr = np.expand_dims(np.array(datetime_array), axis=2)
    seq_x = np.concatenate((dt_arr, seq_x), axis=2)
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
        

    # Train Test Valid split
    X_train, X_sub, y_train, y_sub = train_test_split(seq_x_scaled, y_cluster, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_sub, y_sub, test_size=0.5, random_state=42)

    train_data = X_train
    train_labels = y_train
    # val_data = torch.Tensor(X_val).to(device)
    # val_labels = torch.LongTensor(y_val).to(device)#.float()
    test_data = torch.Tensor(X_test).to(device)
    test_labels = torch.LongTensor(y_test).to(device)#.float()

    """ Apply SMOTE Oversampling to Training data. """
    if smote:
        print(f"Data size before SMOTE: {len(train_labels)}")
        train_reshaped = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        smote = SMOTE(k_neighbors=4, random_state=42)
        train_resampled, train_labels = smote.fit_resample(train_reshaped, train_labels)
        train_data = train_resampled.reshape(train_resampled.shape[0], train_data.shape[1], train_data.shape[2])
        print(f"Data size after SMOTE: {len(train_labels)}")


    """ Apply Undersampling to Training data. """
    if under_sampling:
        print(f"Data size before SMOTE: {len(train_labels)}")
        train_reshaped = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        rus = RandomUnderSampler(random_state=42)
        train_resampled, train_labels = rus.fit_resample(train_reshaped, train_labels)
        train_data = train_resampled.reshape(train_resampled.shape[0], train_data.shape[1], train_data.shape[2])
        print(f"Data size after SMOTE: {len(train_labels)}")

    # Define the loss function with weights
    criterion = nn.CrossEntropyLoss()

    # Basic testing
    if model_name == 'LSTM':
        model = LSTM_Cluster(input_size=seq_x_scaled.shape[2], hidden_size=hidden_layers, num_layers=num_layers, output_size=len(set(y_cluster))).to(device)
    elif model_name == 'GRU':
        model = GRUModel(input_size=seq_x_scaled.shape[2], hidden_size=hidden_layers, num_layers=num_layers, output_size=len(set(y_cluster))).to(device)

    model.load_state_dict(torch.load(os.path.join(path, f'{model_name}_model_seq_us0124_{SEQ_LENGTH}.pth')))
    model.eval()
    return testing(model, criterion, test_data, test_labels, len(set(y_cluster)), x_scaler, plotting)

if __name__ == '__main__':
    train_and_test(model_name = 'LSTM', smote = False, ohe = False, under_sampling = False, remove_min_grp = True)