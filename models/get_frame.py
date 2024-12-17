import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union


path = 'data/'

# Load the level 2 shapefile
gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
y_init, y_label = [], []
with open(os.path.join(path, 'y_mapping.txt'), 'r') as f:
    for line in f:
        y_init.append(int(line.split(':')[0]))
        gid_num = int(line.split(':')[-1].split('.')[0])
        gid_num1, gid_num2 = int(gid_num//100), gid_num%100
        y_label.append(f'JPN.{gid_num1}.{gid_num2}_1')


# Filter the gdf
y_pd = pd.DataFrame({
    'y_id': y_init, #[i for i in range(clust_cnt)],
    'y_map_label': y_label, # JPN.1.7_1
})

y_pd.crs = gdf2.crs
gdf2 = gdf2[gdf2["GID_2"].isin(y_label)]
gdf2 = gdf2.merge(y_pd, left_on='GID_2', right_on='y_map_label')

# Get the frame of the geometry
combined_geom = unary_union(gdf2['geometry'])
min_lon, min_lat, max_lon, max_lat = combined_geom.bounds