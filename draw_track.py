import os
import folium
import numpy as np 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import branca.colormap as cm
from datetime import datetime
from tqdm import tqdm
from folium import plugins, FeatureGroup
from math import atan2, pi

# Function to add arrows
def add_arrows(line_map, line_coordinates, arrow_color='blue', arrow_size=3, arrow_length=1):
    for i in range(len(line_coordinates) - 1):
        origin = line_coordinates[i]
        destination = line_coordinates[i + 1]
        distance = [destination[0] - origin[0], destination[1] - origin[1]]
        arrow_head = [origin[0] + distance[0] * arrow_length, origin[1] + distance[1] * arrow_length]
        line_map.add_child(folium.RegularPolygonMarker(location=arrow_head,
                                                       number_of_sides=3,
                                                       radius=arrow_size,
                                                       fill_color=arrow_color,
                                                       rotation=-90 + atan2(distance[1], distance[0]) * 180 / pi))


# popn_path = 'p-csv/0013/'
# popn_folder = [folder for folder in os.listdir(popn_path)]
# l, p = [], []

# for csv in tqdm(popn_folder):
#     file = pd.read_csv(os.path.join(popn_path, csv), header=None)
#     l.append(file.shape[0])
#     p.append(os.path.join(popn_path, csv))
# path = p[l.index(max(l))]





path = "p-csv/0013/00267754.csv"
data = pd.read_csv(path, header=None)
time_list, track = [], []
time_record = [0 for _ in range(7)]


for i in range(data.shape[0]):
    time = datetime.strptime(data.iloc[i, 3], '%Y-%m-%d %H:%M:%S') 
    if time.day == 1:
        if time.hour == 10 and time.minute == 0:
            time_record[0] = i
        elif time.hour == 16 and time.minute == 0:
            time_record[1] = i
        elif time.hour == 19 and time.minute == 0:
            time_record[2] = i
    elif time.day == 2:
        if time.hour == 0 and time.minute == 0:
            time_record[3] = i
        elif time.hour == 10 and time.minute == 0:
            time_record[4] = i
        elif time.hour == 16 and time.minute == 0:
            time_record[5] = i
        elif time.hour == 19 and time.minute == 0:
            time_record[6] = i
    
    time_list.append(time)
    track.append((data.iloc[i, 4], data.iloc[i, 5]))

points = [[item[1], item[0]] for item in track]
data = [ {"time": i+1, "lat": k, "lon": j} for i, (j, k) in enumerate(track)]

# Create a map object centered on the average coordinates
average_lat = sum(lat for _, lat in track) / len(track)
average_lon = sum(lon for lon, _ in track) / len(track)
m = folium.Map(location=[average_lat, average_lon], tiles="Cartodb Positron", zoom_start=12)



time_intervals = {
    'Go to work 10/1': range(1, time_record[0]+1), # ~10
    'Working hours 10/1': range(time_record[0]+1, time_record[1]+1), # 10~16
    'Get off work 10/1': range(time_record[1]+1, time_record[2]+1), # 16~19
    'Nightlife 10/1': range(time_record[2]+1, time_record[3]+1), # 19~24
    'Go to work 10/2': range(time_record[3]+1, time_record[4]+1), # ~10
    'Working hours 10/2': range(time_record[4]+1, time_record[5]+1), # 10~16
    'Get off work 10/2': range(time_record[5]+1, time_record[6]+1), # 16~19
    'Nightlife 10/2': range(time_record[6]+1, len(time_list)), # 19~24
}

# Colors for different layers
colors = ['blue', 'green', 'orange', 'red', 'yellow', 'lightblue', 'lightgreen', 'purple']

# Create layers
for (layer_name, time_range), color in zip(time_intervals.items(), colors):
    layer = FeatureGroup(name=layer_name)
    linear = cm.LinearColormap(
        ['green', 'yellow', 'orange', 'red'],
        vmin=1, vmax=len(time_range)+1
    )

    for point in data:
        if point["time"] in time_range:
            folium.CircleMarker(
                location=[point["lat"], point["lon"]],
                radius=1,
                color=linear(point["time"] - time_range[0]),
                fill=True,
                fill_color=linear(point["time"])
            ).add_to(layer)
    layer.add_to(m)    
    

# Add layer control
folium.LayerControl().add_to(m)
linear.add_to(m)

# # Plot each point
# for point in data:
#     folium.CircleMarker(
#         location=[point["lat"], point["lon"]],
#         radius=1,
#         color=linear(point["time"]),
#         fill=True,
#         fill_color=linear(point["time"])
#     ).add_to(m)



# # Add arrows to the map
# for i in range(len(points) - 1):
#     folium.PolyLine([points[i], points[i + 1]], color='black', weight=2, opacity=1).add_to(m)
#     add_arrows(m, [points[i], points[i + 1]])

# # colormap = plt.cm.get_cmap('tab20', 20)  # 'tab20' colormap with 20 distinct colors
# for time, (lon, lat) in zip(time_list, track):
#     if time.minute == 0:
#         folium.CircleMarker(location=[lat, lon], radius=1, color='red').add_to(m)
#         folium.Marker(
#             location=[lat, lon],
#             popup=time,  # Popup text
#             icon=folium.Icon(icon='info-sign', color='green')
#         ).add_to(m)

m.save('visualization/track_map_with_arrows.html')