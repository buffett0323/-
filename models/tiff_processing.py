"""Range from 138-141, 34-37"""
import rasterio
import pandas as pd
import numpy as np
import os
from params import SEQ_LENGTH

lat_range = [34, 35, 36]
lon_range = [138, 139, 140]
dataset = dict()
SHP = 12000

for l1 in lat_range:
    for l2 in lon_range :
        if l1 == 34 and l2 == 140:
            continue
        with rasterio.open(f'../Japan_Data/JAXA_HRLULC_Japan_v23.12/LC_N{l1}E{l2}.tif') as src:
            # Read the first band
            first_band = src.read(1)
            SHP = src.width
            dataset[f'N{l1}E{l2}'] = first_band
            # np.save(os.path.join("data/", f'N{l1}E{l2}'), first_band)
 

def lat_lon_to_indices(lon, lat):
    if int(np.floor(lat)) not in lat_range or int(np.floor(lon)) not in lon_range: 
        return -9999
    
    # Normalize the latitude and longitude values
    normalized_lat = np.ceil(lat) - lat
    normalized_lon = lon - np.floor(lon)

    # Convert to array indices
    lat_idx = int(normalized_lat * (SHP - 1))
    lon_idx = int(normalized_lon * (SHP - 1))
    return dataset[f'N{int(np.floor(lat))}E{int(np.floor(lon))}'][lat_idx][lon_idx]
    # return lat_idx, lon_idx


# Main
if __name__ == '__main__':
    seq_data = np.load(f'data/trans_{SEQ_LENGTH}.npy', allow_pickle=True)
    seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
    ll_trans = []
    for i in range(seq_data.shape[0]):
        for j in range(seq_data.shape[1]):
            tmp = list(seq_data[i,j,:])
            ll = lat_lon_to_indices(seq_data[i, j, 6], seq_data[i, j, 7])
            if ll == -9999:
                print("Error"); break
            else:
                ll_trans.append(ll)
    
    new_column = np.array(ll_trans).reshape((seq_data.shape[0] * seq_data.shape[1], 1)) # 2D
    seq_2d = np.concatenate((seq_2d, new_column), axis=1)
    
    
    last_col = seq_2d[:, -1].copy()
    seq_2d[:, 10:] = seq_2d[:, 9:-1]
    seq_2d[:, 9] = last_col
    
    seq_3d = seq_2d.reshape((int(seq_2d.shape[0]/SEQ_LENGTH), SEQ_LENGTH, seq_2d.shape[1]))
    np.save(f'data/trans_new_{SEQ_LENGTH}.npy', seq_3d)
    print(f'Finish storing SEQ:{SEQ_LENGTH}!')


