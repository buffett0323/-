import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from params import SEQ_LENGTH

plotting_method = "scatter"
seq_data = np.load(f"data/seqRegion_{SEQ_LENGTH}.npy", allow_pickle=True)
seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
max_lon, min_lon = max(seq_2d[:,6].tolist()), min(seq_2d[:,6].tolist())
max_lat, min_lat = max(seq_2d[:,7].tolist()), min(seq_2d[:,7].tolist())


l1_label_num, l1_label = [], []
with open(f"data/l1_label_{SEQ_LENGTH}.txt", 'r') as file:
    for line in file:
        l1_label_num.append(int(line.split(':')[0]))
        l1_label.append(line.split(':')[-1].split('\n')[0])

region_total = [int(seq_data[i,-1,-1]) for i in range(seq_data.shape[0])]
region_count = [region_total.count(i) for i in l1_label_num]

reg_df = pd.DataFrame({
    'y_id': l1_label_num,
    'label': l1_label,
    'count': region_count
})


# Plotting
if plotting_method == "area":
    fig, ax = plt.subplots()
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')

    # Filter
    gdf1 = gdf1[gdf1["NAME_1"].isin(l1_label)]
    gdf1 = gdf1.merge(reg_df, left_on='NAME_1', right_on='label')

    gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2')
    gdf1.boundary.plot(ax=ax, color='red', label='Level 1')
    gdf1.plot(column='count', ax=ax, legend=True, cmap='OrRd')


    for idx, row in gdf1.iterrows():
        centroid = row.geometry.centroid # Get the centroid of the polygon
        id_clust = ''
        if row.NAME_1 in l1_label:
            id_clust = str(l1_label.index(row.NAME_1))
        
        # Annotate the plot
        # plt.annotate(text=row['count'], xy=(centroid.x, centroid.y), xytext=(3, 3), textcoords="offset points")

    longitude_range = [np.floor(min_lon), np.ceil(max_lon)]
    latitude_range = [34.8, np.ceil(max_lat)]
    ax.set_xlim(longitude_range)
    ax.set_ylim(latitude_range)
    ax.legend()

    plt.title('Administrative District Count')
    plt.savefig(f'visualization/region_count.png')
    plt.show()

elif plotting_method == "scatter":
    fig, ax = plt.subplots()
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')

    # Filter
    gdf1 = gdf1[gdf1["NAME_1"].isin(l1_label)]
    gdf1 = gdf1.merge(reg_df, left_on='NAME_1', right_on='label')

    gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2')
    gdf1.boundary.plot(ax=ax, color='black', label='Level 1')

    for i in range(seq_data.shape[0]):
        ax.scatter(seq_data[i, -1, 6], seq_data[i, -1, 7], c='r', s=5)


    longitude_range = [np.floor(min_lon), np.ceil(max_lon)]
    latitude_range = [34.8, np.ceil(max_lat)]
    ax.set_xlim(longitude_range)
    ax.set_ylim(latitude_range)
    ax.legend()

    plt.title('Administrative District Count')
    plt.savefig(f'visualization/region_count_scatter.png')
    plt.show()

    
    