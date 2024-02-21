import os
import folium
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
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


"""Using folium to generate the map"""
gdf2_json = json.loads(gdf2.to_json())
m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=10)#, tiles='CartoDB positron')



target_gid_2 = 176
adjacent_gids = [205, 185, 189, 162, 213, 193]


# Function to create a tooltip for each feature in the GeoJSON
def create_tooltip(feature):
    gid_2 = feature['properties']['GID_2']
    y_id = y_label.index(gid_2) if gid_2 in y_label else None
    return str(y_init[y_id]) if y_id is not None else ''

def get_color(gid_2):
    if gid_2 == target_gid_2:        
        return 'green'  # Target polygon color
    elif gid_2 in adjacent_gids:
        return 'orange'  # Adjacent polygons color
    else:
        return 'lightblue'  # Default color for other polygons




# Add polygons to the map
folium.GeoJson(
    gdf2_json,
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties']['y_id']),
        'color': 'black',  # Outline color
        'weight': 2,  # Outline weight
        'fillOpacity': 0.7,  # Adjust fill opacity
        'opacity': 0.5, 
        'dashArray': '5, 5'  # Outline dash pattern
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['y_id'],  # Adjust according to your GeoJSON properties
        aliases=['Y Label:'],  # Adjust tooltip label as needed
        style=('background-color: white; color: black;')
    ),
).add_to(m)

# Add layer control to toggle between layers
folium.LayerControl().add_to(m)

# Custom Legend HTML
legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 130px; 
     border:3px solid rgba(0,0,0,0.5); z-index:9999; font-size:16px; 
     background-color: rgba(255,255,255,0.8); padding: 10px; 
     box-shadow: 3px 3px 5px rgba(0,0,0,0.5);">
     &nbsp; Target &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i><br>
     &nbsp; Neighbors &nbsp; <i class="fa fa-map-marker fa-2x" style="color:orange"></i><br>
     &nbsp; Others &nbsp; <i class="fa fa-map-marker fa-2x" style="color:lightblue"></i>
     </div>
     '''

# Add the Legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Save map to an HTML file
m.save('visualization/adj_acc_plotting.html')

