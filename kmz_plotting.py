from lxml import etree
from PIL import Image, ImageDraw
from params import SEQ_LENGTH
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import zipfile
import os

# extract the kmz file 
kmz_file = '../Japan_Data/landuse_Tokyo/tokyo2000.kmz'
extract_folder = 'kmz_2000'

with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


# Load and parse the KML file
kml_file = 'kmz_2000/doc.kml'
tree = etree.parse(kml_file)
root = tree.getroot()

# Define the namespace map
ns_map = {'kml': 'http://www.opengis.net/kml/2.2'}

# Find the GroundOverlay element
ground_overlay = root.find('.//kml:GroundOverlay', namespaces=ns_map)

# Extract the image path from the Icon element
image_path = ground_overlay.find('.//kml:Icon/kml:href', namespaces=ns_map).text

# Extract the LatLonBox coordinates
lat_lon_box = ground_overlay.find('.//kml:LatLonBox', namespaces=ns_map)
north = lat_lon_box.find('kml:north', namespaces=ns_map).text
south = lat_lon_box.find('kml:south', namespaces=ns_map).text
east = lat_lon_box.find('kml:east', namespaces=ns_map).text
west = lat_lon_box.find('kml:west', namespaces=ns_map).text

# Open the image file and process
with Image.open(os.path.join(extract_folder, image_path)) as img:
    img = img.convert('RGB')
    img_array = np.array(img)
    red_channel = img_array[:, :, 0]  # Red channel
    green_channel = img_array[:, :, 1]  # Green channel
    blue_channel = img_array[:, :, 2]  # Blue channel

image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

# Plot the image
fig, ax = plt.subplots()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())

# Set the extent to the geographical bounds of the image
ax.set_extent([float(west), float(east), float(south), float(north)], crs=ccrs.PlateCarree())
ax.imshow(image, origin='upper', extent=[float(west), float(east), float(south), float(north)], transform=ccrs.PlateCarree())
# ax.coastlines()
ax.gridlines(draw_labels=True)
ax.axis('off')  # Turn off axis numbers and ticks
ax.set_xticks([], crs=ccrs.PlateCarree())
ax.set_yticks([], crs=ccrs.PlateCarree())
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')
gdf1.boundary.plot(ax=ax, color='gray', label='Level 1')


# plotting dots
seq_data = np.load(f"data/seqRegion_{SEQ_LENGTH}.npy", allow_pickle=True)
for i in range(seq_data.shape[0]):
    item = seq_data[i, :, :]
    for id in range(item.shape[0]):
        """ 8 for sea, 6 for water """
        if int(item[id, -2]) == 6: 
            lon, lat = item[id, 6], item[id, 7]
            ax.scatter(lon, lat, c='black', marker='x', s=5)


plt.xticks([])
plt.yticks([])
plt.show()