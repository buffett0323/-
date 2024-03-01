import folium
from get_frame import min_lon, min_lat, max_lon, max_lat


# Initialize the map centered at the average location of the points
latitude_list = [35.6703, 35.6711, 35.6735, 35.6748]
longitude_list = [139.738, 139.7428, 139.759, 139.7596]

m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], tiles="Cartodb Positron", zoom_start=12)

# Define the properties of the circle markers
circle_properties = {
    'radius': 5,
    'color': '#grey',
    'fill': True,
    'fill_color': '#grey',
    'fill_opacity': 0.7,
}

# Add circle markers to the map
for lat, long in zip(latitude_list, longitude_list):
    folium.CircleMarker(location=[lat, long], **circle_properties).add_to(m)

# Display the map
m.save('visualization/track_testing.html')