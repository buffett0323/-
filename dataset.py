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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from test_visualization import visualize_cut_result, visualize_admin_result
from params import device


# General dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.data[idx])).to(device), torch.LongTensor(np.array([self.labels[idx]])).to(device)



# Split the sequence and the static features
class SplitDataset(Dataset):
    def __init__(self, sequences, static_features, labels):
        """
        sequences: A tensor containing all sequences (e.g., time-series data)
        static_features: A tensor containing all static features
        labels: A tensor containing the labels
        """
        self.sequences = sequences
        self.static_features = static_features
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.sequences[idx])).to(device), \
                torch.FloatTensor(np.array(self.static_features[idx])).to(device), \
                torch.LongTensor(np.array([self.labels[idx]])).to(device)


# Dataset for regression
class RegDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(np.array(self.data[idx])).to(device), torch.FloatTensor(np.array([self.labels[idx]])).to(device)
