import numpy as np
from params import SEQ_LENGTH, occupation, trip_purpose, transport_type
from prettytable import PrettyTable
import geopandas as gpd
import matplotlib.pyplot as plt

seq_data = np.load(f"data/seqRegion_{SEQ_LENGTH}.npy", allow_pickle=True)
work_list = []
for i in range(seq_data.shape[0]):
    work_list.append(seq_data[i, 0, 3])
    
work_name = []
work_count = []
for w in set(work_list):
    work_name.append(occupation[w])
    work_count.append(work_list.count(w))

list_to_sort = work_count  
list_to_reorder = work_name

# Get the sorted indices based on list_to_sort
sorted_indices = sorted(range(len(list_to_sort)), key=list_to_sort.__getitem__, reverse=True)

# Use the sorted indices to sort both lists
sorted_list_to_sort = [list_to_sort[i] for i in sorted_indices]
sorted_list_to_reorder = [list_to_reorder[i] for i in sorted_indices]

cnt = 0
for i, j in zip(sorted_list_to_sort, sorted_list_to_reorder):
    cnt += 1
    print(i, j)
    if cnt >= 4:
        break
