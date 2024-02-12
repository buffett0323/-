import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import SEQ_LENGTH, CUT
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler
LOOKBACK = SEQ_LENGTH - 1

def region2id(y, cut):
    thres = float(1/cut)
    thres_range = [float(i * thres) for i in range(cut)]
    id = 0
    for i, thres in enumerate(thres_range):
        if y >= thres:
            id = i
    return id

seq_data = np.load(f'data/seqRegion_{SEQ_LENGTH}.npy', allow_pickle=True) # (6, 770400, 8)
seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
max_lon, min_lon = max(seq_2d[:,6].tolist()), min(seq_2d[:,6].tolist())
max_lat, min_lat = max(seq_2d[:,7].tolist()), min(seq_2d[:,7].tolist())

seq_data = np.transpose(seq_data, (1, 0, 2))
seq_x = seq_data[:LOOKBACK, :, 1: -1].astype(np.float64) # Temporarily remove 0 (the time)
seq_y = seq_data[LOOKBACK, :, [6, 7]].astype(np.float64) # Predict the box
seq_x = np.transpose(seq_x, (1, 0, 2))
seq_y = np.transpose(seq_y, (1, 0))

scaler = MinMaxScaler()
seq_y1 = seq_y[:, 0].reshape(-1, 1)
seq_y2 = seq_y[:, 1].reshape(-1, 1)
seq_y1_scale = scaler.fit_transform(seq_y1) 
seq_y2_scale = scaler.fit_transform(seq_y2)
y_cluster = [region2id(y1, CUT) * CUT + region2id(y2, CUT) for y1, y2 in zip(seq_y1_scale, seq_y2_scale)]
y_pd = pd.DataFrame({
    'y_id': [i for i in range(CUT*CUT)],
    'count': [y_cluster.count(i) for i in range(CUT*CUT)]
})


# Calculate the step size for each grid cell
lon_step = (max_lon - min_lon) / CUT
lat_step = (max_lat - min_lat) / CUT

# Create the polygons for each grid cell
polygons, poly_id = [], []
for i in range(CUT):
    for j in range(CUT):
        lon_start = min_lon + i * lon_step
        lat_start = min_lat + j * lat_step
        polygon = Polygon([
            (lon_start, lat_start),
            (lon_start + lon_step, lat_start),
            (lon_start + lon_step, lat_start + lat_step),
            (lon_start, lat_start + lat_step),
            (lon_start, lat_start)  # Close the polygon
        ])
        polygons.append(polygon)
        poly_id.append(i * CUT + j)

gdf = gpd.GeoDataFrame({
    'geometry': polygons,
    'poly_id': poly_id
    })
gdf = gdf.merge(y_pd, left_on='poly_id', right_on='y_id')


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

fig, ax = plt.subplots()
gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')

# Filter
gdf1 = gdf1[gdf1["NAME_1"].isin(l1_label)]
gdf1 = gdf1.merge(reg_df, left_on='NAME_1', right_on='label')

gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2')
gdf1.boundary.plot(ax=ax, color='black', label='Level 1', alpha=0.2)
gdf.plot(column='count', ax=ax, legend=True, cmap='OrRd')


for i in range(1, CUT):
    x_tmp = min_lon + i * (max_lon - min_lon) / CUT
    y_tmp = min_lat + i * (max_lat - min_lat) / CUT
    ax.plot([x_tmp, x_tmp], [min_lat, max_lat], color='gray', linestyle='--', alpha=0.2)
    ax.plot([min_lon, max_lon], [y_tmp, y_tmp], color='gray', linestyle='--', alpha=0.2)

ax.plot([min_lon, min_lon], [min_lat, max_lat], color='gray', linestyle='-')
ax.plot([max_lon, max_lon], [min_lat, max_lat], color='gray', linestyle='-')
ax.plot([min_lon, max_lon], [min_lat, min_lat], color='gray', linestyle='-')
ax.plot([min_lon, max_lon], [max_lat, max_lat], color='gray', linestyle='-')


for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid # Get the centroid of the polygon

    # Annotate the plot
    plt.annotate(text=row['count'], xy=(centroid.x, centroid.y), xytext=(3,3), ha='center', va='center', fontsize=6, textcoords="offset points")

longitude_range = [round(min_lon, 1) - 0.3 , round(max_lon, 1) + 0.3]
latitude_range = [round(min_lat, 1) - 0.3, round(max_lat, 1) + 0.3]
ax.set_xlim(longitude_range)
ax.set_ylim(latitude_range)
ax.legend()

plt.title(f'Japan District with {CUT}x{CUT} Cut Count')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.savefig(f'visualization/visual_{CUT}x{CUT}_count.jpg')
plt.show()