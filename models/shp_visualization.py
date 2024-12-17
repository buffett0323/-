"""Shp file processing with Y value"""
import os
import folium
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from params import CUT, SEQ_LENGTH
from get_frame import min_lon, min_lat, max_lon, max_lat

path = 'data/'

# Load the level 2 shapefile
gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')


""" Plot the shapefile """
fig, ax = plt.subplots()
gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2')
gdf1.boundary.plot(ax=ax, color='black', label='Level 1', alpha=0.2)

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
gdf2 = gdf2.to_crs('EPSG:4326')


# Project to m^2 and km^2
gdf_projected = gdf2.to_crs('EPSG:6677')

# Calculate the area, with m^2, km^2
gdf_projected['area_m2'] = gdf_projected['geometry'].area
gdf_projected['area_km2'] = gdf_projected['area_m2'] / 1e6

average_area_m2 = gdf_projected['area_m2'].mean()
std_dev_area_m2 = gdf_projected['area_m2'].std()
average_area_km2 = gdf_projected['area_km2'].mean()
std_dev_area_km2 = gdf_projected['area_km2'].std()

print("Average Area:", round(average_area_m2, 4), "m^2")
print("Standard Deviation of Area:", round(std_dev_area_m2, 4), "m^2")
print("Average Area:", round(average_area_km2, 4), "km^2")
print("Standard Deviation of Area:", round(std_dev_area_km2, 4), "km^2")


"""Using folium to generate the map"""
gdf2_json = json.loads(gdf2.to_json())
m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=10,  tiles='CartoDB positron')

# Function to create a tooltip for each feature in the GeoJSON
def create_tooltip(feature):
    gid_2 = feature['properties']['GID_2']
    y_id = y_label.index(gid_2) if gid_2 in y_label else None
    return str(y_init[y_id]) if y_id is not None else ''

# Add polygons to the map
folium.GeoJson(
    gdf2_json,
    style_function=lambda feature: {
        'fillColor': 'lightblue',
        'color': 'blue',
        'weight': 2,
        'dashArray': '5, 5'
    },
    tooltip=folium.GeoJsonTooltip(fields=['y_id'], aliases=['Y Label:'], style=('background-color: white; color: black;')),
).add_to(m)

# Add layer control to toggle between layers
folium.LayerControl().add_to(m)

# Save map to an HTML file
m.save('visualization/y_level2.html')




# for idx, row in gdf2.iterrows():
#     # Get the centroid of the polygon
#     centroid = row.geometry.centroid
#     id_clust = ''
#     if row.GID_2 in y_label:
#         y_id = y_label.index(row.GID_2)
#         id_clust = str(y_init[y_id])
    
#     # Annotate the plot
#     plt.annotate(text=id_clust, xy=(centroid.x, centroid.y), xytext=(1, 1), textcoords="offset points")


# for i in range(1, CUT):
#     x_tmp = min_lon + i * (max_lon - min_lon) / CUT
#     y_tmp = min_lat + i * (max_lat - min_lat) / CUT
#     ax.plot([x_tmp, x_tmp], [min_lat, max_lat], color='blue', linestyle='--')
#     ax.plot([min_lon, max_lon], [y_tmp, y_tmp], color='blue', linestyle='--')

# ax.plot([min_lon, min_lon], [min_lat, max_lat], color='black', linestyle='-')
# ax.plot([max_lon, max_lon], [min_lat, max_lat], color='black', linestyle='-')
# ax.plot([min_lon, max_lon], [min_lat, min_lat], color='black', linestyle='-')
# ax.plot([min_lon, max_lon], [max_lat, max_lat], color='black', linestyle='-')
# # ax.axhline(y=y_tmp, color='blue', linestyle='-')


# # Define the latitude and longitude range
# longitude_range = [min_lon, max_lon]
# latitude_range = [min_lat, max_lat]
# ax.set_xlim(longitude_range)
# ax.set_ylim(latitude_range)
# ax.legend()

# # Add additional plot formatting here as needed
# plt.title(f'Japan District Scatter Plot')
# plt.xlabel('longitude')
# plt.ylabel('latitude')

# # plt.savefig(f'visualization/visual_{CUT}x{CUT}.png')
# plt.show()