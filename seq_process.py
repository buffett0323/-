# New editing in 1/2
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import multiprocessing
from params import SEQ_LENGTH
from shapely.geometry import Point
# from tqdm import tqdm
from datetime import datetime

# init settings
dir_txt = 'kmz_2000/dir_info.txt'
legend_array = np.load('kmz_2000/legend.npy')

with open(dir_txt, 'r') as f:
    lines = f.readlines()
    north = float(lines[0].split(" ")[1])
    south = float(lines[1].split(" ")[1])
    east = float(lines[2].split(" ")[1])
    west = float(lines[3].split(" ")[1])

def lat_lon_to_indices(lon, lat):
    lat_range = north - south
    lon_range = east - west

    # Normalize the latitude and longitude values
    normalized_lat = (north - lat) / lat_range
    normalized_lon = (lon - west) / lon_range

    # Convert to array indices
    lat_idx = int(round(normalized_lat * (legend_array.shape[0] - 1), 0))
    lon_idx = int(round(normalized_lon * (legend_array.shape[1] - 1), 0))

    return lat_idx, lon_idx

def worker(popn_path, folder):
    folder_path = os.path.join(popn_path, folder)
    seq_data = []
    csv_list = [pd.read_csv(os.path.join(folder_path, csv), header=None) for csv in os.listdir(folder_path)] # if csv.endswith("01.csv")

    # Read csv file
    for csv_pd in csv_list:
        # Get Sequence ID based on the trip purpose change
        from_id = seq_indices(csv_pd.iloc[:,10].tolist())
        
        # Threshold for sequence length
        if len(from_id) >= SEQ_LENGTH:
            tmp_data = csv_pd.iloc[from_id, [3,6,7,9,10,13,4,5]]
            clust_idx = [lat_lon_to_indices(tmp_data.iloc[i, 6], tmp_data.iloc[i, -1]) for i in range(tmp_data.shape[0])]
            clust_list = []
            add = True
            for item in clust_idx:
                if (item[0] < legend_array.shape[0] and item[1] < legend_array.shape[1]):
                    clust_list.append(int(legend_array[item[0], item[1]]))
                else:
                    add = False
            # tmp_data["clust"] = [int(legend_array[item[0], item[1]]) for item in clust_idx if (item[0] < legend_array.shape[0] and item[1] < legend_array.shape[1]) else -999999]
            
            if add:
                tmp_data["clust"] = clust_list
                for i in range(len(from_id)- SEQ_LENGTH + 1):
                    seq_data.append(tmp_data.iloc[list(range(i, i+SEQ_LENGTH)), :])   
    
    return np.array(seq_data)

def seq_indices(lst):
    change_indices = [0] 
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            change_indices.append(i)
    return change_indices


# Main
if __name__ == '__main__':
    popn_path = '../POPN_store/'
    popn_folder = [f"{folder}/p-csv" for folder in os.listdir(popn_path)]
    csv_list = []
    seq_data = []
    num_cores = multiprocessing.cpu_count()
    
    for i, popn_str in enumerate(popn_folder):
        st = int(str(popn_str.split('/')[0])[-2:]) - 26
        if st < 10:
            popn_folder[i] = popn_str + "/" + f"000{str(st)}" 
        else:
            popn_folder[i] = popn_str + "/" + f"00{str(st)}"   
    
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(worker, [(popn_path, folder) for folder in popn_folder])

    seq_data = np.concatenate(results, axis=0)
    
    # Load the level 2 shapefile
    gdf2 = gpd.read_file('../gadm41_JPN_shp/gadm41_JPN_2.shp')
    gdf1 = gpd.read_file('../gadm41_JPN_shp/gadm41_JPN_1.shp')
    # gdf0 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_0.shp')
    seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
    points_df = pd.DataFrame(seq_2d[:, 6:8])
    points_df.columns = ['longitude', 'latitude']

    geometry = [Point(x, y) for x, y in zip(seq_2d[:,6], seq_2d[:,7])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry)

    # Set the same CRS as the polygons GeoDataFrame
    points_gdf.crs = gdf2.crs # points_gdf.crs = gdf1.crs
    joined_gdf2 = gpd.sjoin(points_gdf, gdf2, how='left', op='within')
    joined_res2 = joined_gdf2[['GID_1', 'GID_2']] #.to_numpy()
    
    # Transform y labels to numbers and record it
    gid1, gid2 = [], []
    for i in range(joined_res2.shape[0]):
        if str(joined_res2.iloc[i, 0]) != 'nan':
            gid1.append(int(str(joined_res2.iloc[i, 0]).split('_')[0].split('.')[-1]))
        else:
            gid1.append(-99)
        if str(joined_res2.iloc[i, 1]) != 'nan':
            gid2.append(int(str(joined_res2.iloc[i, 1]).split('_')[0].split('.')[-1]))
        else:
            gid2.append(-99)

       
    # Filtering nan values
    seq_2d = np.concatenate((seq_2d, np.array(gid1).reshape(-1, 1)), axis=1)
    seq_2d = np.concatenate((seq_2d, np.array(gid2).reshape(-1, 1)), axis=1)
    seq_3d = seq_2d.reshape((int(seq_2d.shape[0]/SEQ_LENGTH), SEQ_LENGTH, seq_2d.shape[1]))  
    
    mask = [False if -99.0 in list(item) else True for item in seq_3d[:,:,-2]]
    for i in range(len(mask)):
        if mask[i]:
            if -99.0 in list(seq_3d[i, :, -1]):
                mask[i] = False
  

    seq_3d_filt = seq_3d[mask]
    seq_2d = seq_3d_filt.reshape(seq_3d_filt.shape[0] * seq_3d_filt.shape[1], seq_3d_filt.shape[2])
    print('False count: ', mask.count(False), seq_3d.shape, seq_3d_filt.shape)  

    # Data correction
    for i in range(seq_2d.shape[0]):
        if seq_2d[i, 8] == 6 or seq_2d[i, 8] == 8:
            if datetime.strptime(seq_2d[i, 0], '%Y-%m-%d %H:%M:%S').hour < 17:
                seq_2d[i, 8] = 4
            else:
                seq_2d[i, 8] = 5

    seq_3d_filt = seq_2d.reshape(seq_3d_filt.shape[0], seq_3d_filt.shape[1], seq_3d_filt.shape[2])

    np.save(f'data/newData_{SEQ_LENGTH}.npy', seq_3d_filt)
    print('Finish storing!')


