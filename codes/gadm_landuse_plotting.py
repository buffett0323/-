import os
import zipfile
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point
from shapely.ops import unary_union
from get_frame import min_lon, min_lat, max_lon, max_lat
from lxml import etree
from PIL import Image, ImageDraw
import cartopy.crs as ccrs

path = 'data/'

# Load the level 2 shapefile
gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')


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


# Create a figure and a single subplot with Cartopy's projection
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot GeoDataFrame boundaries on the same axis object with Cartopy projection
gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2', transform=ccrs.PlateCarree())
gdf1.boundary.plot(ax=ax, color='black', label='Level 1', alpha=0.4, transform=ccrs.PlateCarree())

# Set the geographical extent of the plot to cover the area of interest
ax.set_extent([139, 140.25, 35, 36.3], crs=ccrs.PlateCarree())

# Plot the image within the specified extent and with the correct transformation
ax.imshow(image, origin='upper', extent=[float(west), float(east), float(south), float(north)], transform=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)



# Create custom legend entries
colors = {"Forest": "green", "Grassland": "lightgreen", "Rice Field": "yellow", "Agricultural": "orange",
    "Industrial": "purple", "Urban area": "red", "Water land": "lightblue", "Others": "grey", "Ocean": "blue"}
lines = {'Level 1': 'black', 'Level 2': 'lightblue'}
patch_handles = [mpatches.Patch(color=color, label=category) for category, color in colors.items()]
line_handles = [mlines.Line2D([], [], color=color, label=label, linewidth=2) for label, color in lines.items()]

# Show both patch and line legends
legend_handles = patch_handles + line_handles
ax.legend(handles=legend_handles, loc='lower left', ncol=2)


# Show the plot
plt.title('GADM & Landuse', fontsize=15, fontweight='bold')
plt.text(0.5, -0.1, "Longitude", va='bottom', ha='center',
         rotation='horizontal', rotation_mode='anchor',
         transform=ax.transAxes)
plt.text(-0.15, 0.5, "Latitude", va='bottom', ha='center',
         rotation='vertical', rotation_mode='anchor',
         transform=ax.transAxes)

plt.savefig('visualization/gadm_landuse.png', dpi=300)
plt.show()
