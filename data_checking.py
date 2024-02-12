import numpy as np
from params import SEQ_LENGTH, occupation, trip_purpose, transport_type
from prettytable import PrettyTable
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

seq_data = np.load(f"data/seqRegion_{SEQ_LENGTH}.npy", allow_pickle=True)
seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
max_lon, min_lon = max(seq_2d[:,6].tolist()), min(seq_2d[:,6].tolist())
max_lat, min_lat = max(seq_2d[:,7].tolist()), min(seq_2d[:,7].tolist())
# seq_data = np.load(f"data/seqRegion_{SEQ_LENGTH}.npy", allow_pickle=True)
gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')

clust_count = []
for i in seq_data[:, :, -2]:
    clust_count.append(i[0])

label_list = ['森林', '草原', '田', '農地', '工業用地', '宅地', 
              '水域', '其他', '海']
for i in range(len(label_list)):
    print(label_list[i], clust_count.count(i))

# legend_array = np.load('kmz_2000/legend.npy', allow_pickle=True)
# print(legend_array)


"""PLOTTING"""
# Plot the shapefile
fig, ax = plt.subplots()
gdf1.boundary.plot(ax=ax, color='black', label='Level 1')

# Define the latitude and longitude range
longitude_range = [np.floor(min_lon), np.ceil(max_lon)]
latitude_range = [np.floor(min_lat), np.ceil(max_lat)]
ax.set_xlim(longitude_range)
ax.set_ylim(latitude_range)
ax.legend()

table = PrettyTable()
table.field_names = ['Occupation', 'Trip Purpose', 'Transport']

for i in range(seq_data.shape[0]):
    item = seq_data[i, :, :]
    for id in range(item.shape[0]):
        if int(item[id, -2]) == 8: 
            table.add_row([occupation[item[id, 3]], trip_purpose[item[id, 4]], transport_type[item[id, 5]]])
            lon, lat = item[id, 6], item[id, 7]
            ax.scatter(lon, lat, c='r', s=5)
        
print(table)

# Add additional plot formatting here as needed
plt.title(f'Check')
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.savefig(f'visualization/check.png')
plt.show()