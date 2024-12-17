from lxml import etree
from PIL import Image, ImageDraw
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

# Processing
dir_info = [f"North: {north} ", f"South: {south} ", 
            f"East: {east} ", f"West: {west} "]

with open(os.path.join(extract_folder, "dir_info.txt"), 'w') as file:
    for line in dir_info:
        file.write(line + "\n")

# Labeling
color_list = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (197, 0, 255), (255, 170, 0), 
              (211, 255, 190), (56, 168, 0), (0, 197, 255), (130, 130, 130), (0, 112, 255)]
color_label = [-999, 5, 2, 4, 3, 1, 0, 6, 7, 8]
legend_array = np.zeros((img_array.shape[0], img_array.shape[1]))

for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        tmp_col = (red_channel[i,j], green_channel[i,j], blue_channel[i,j]) 
        c_id = color_list.index(tmp_col)
        legend_array[i, j] = int(color_label[c_id])
           
print(img_array.shape)

# Export
np.save(os.path.join(extract_folder, "legend.npy"), legend_array)

# # Create a new image with white background
# img = Image.new('RGB', (500, 50 * len(color_list)), (255, 255, 255))
# draw = ImageDraw.Draw(img)

# # Draw each color as a rectangle
# for i, color in enumerate(color_list):
#     draw.rectangle([0, i*50, 500, (i+1)*50], fill=color)

# # Save or show the image
# img.save(f'{extract_folder}/files/color_label.jpg')
# img.show()





   