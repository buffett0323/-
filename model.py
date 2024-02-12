import torch 
import torch.nn as nn
from params import device, SEQ_LENGTH


# Basic LSTM Model
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


# Basic GRU Model
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


# Hybrid LSTM Model
class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, static_feature_size, fc_layer=64, drop_prob=0):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob).to(device)
        self.bn = nn.BatchNorm1d(hidden_size).to(device)  # Batch normalization layer
        self.fc1 = nn.Linear(hidden_size + static_feature_size, fc_layer).to(device)
        self.fc2 = nn.Linear(fc_layer, output_size).to(device)

    def forward(self, sequence, static_features): # Sequence, Static
        lstm_out, _ = self.lstm(sequence)
        # lstm_out = self.bn(lstm_out.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        lstm_out = lstm_out[:, -1, :]

        # Concatenate LSTM output with static features
        combined = torch.cat((lstm_out, static_features), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    

# Hybrid GRU Model
class HybridGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, static_feature_size, fc_layer=64, drop_prob=0.2):
        super(HybridGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob).to(device)
        self.bn = nn.BatchNorm1d(hidden_size).to(device)  # Batch normalization layer
        self.fc1 = nn.Linear(hidden_size + static_feature_size, fc_layer).to(device)
        self.fc2 = nn.Linear(fc_layer, output_size).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, sequence, static_features):  # Sequence, Static
        gru_out, _ = self.gru(sequence)
        gru_out = self.bn(gru_out.permute(0, 2, 1)).permute(0, 2, 1)  # Apply batch normalization
        gru_out = gru_out[:, -1, :]  # Get the output from the last time step

        # Concatenate GRU output with static features
        combined = torch.cat((gru_out, static_features), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    


# Hybrid and Weighted
class HybridWeightedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, static_feature_size, sequence_length, fc_layer=64, weight=3, linear=True, drop_prob=0.2):
        super(HybridWeightedLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.weight = weight
        self.linear = linear
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob).to(device)
        self.fc1 = nn.Linear(hidden_size * sequence_length + static_feature_size, fc_layer).to(device)
        self.fc2 = nn.Linear(fc_layer, output_size)

    def forward(self, sequence, static_features): # Sequence, Static
        lstm_out, _ = self.lstm(sequence)

        # Apply linear weights or exponential weights to the LSTM output
        if self.linear: # Linear
            weights = torch.linspace(1, self.weight, lstm_out.shape[1]).to(sequence.device)
        else: # Exponential 
            weights = torch.pow(torch.full((lstm_out.shape[1],), self.weight).to(sequence.device), torch.arange(lstm_out.shape[1]).to(sequence.device))
       
        weighted_lstm_out = lstm_out * weights.view(1, -1, 1)
        weighted_lstm_out_flat = weighted_lstm_out.reshape(weighted_lstm_out.size(0), -1)

        # Concatenate LSTM output with static features
        combined = torch.cat((weighted_lstm_out_flat, static_features), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    
    

# Basic Regression Model
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output layer for longitude and latitude

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        output = self.fc(last_time_step_out)
        return output
