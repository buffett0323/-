""" Used for gm-jpn-all_u_2_2 dataset """
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

def plot_shapefiles(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder not found")
        return

    # Initialize a plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".shp") and filename == "raill_jpn.shp":
            file_path = os.path.join(folder_path, filename)
            # Read the shapefile
            gdf = gpd.read_file(file_path)

            # Plot each shapefile with a different color and label
            gdf.plot(ax=ax, alpha=0.5, edgecolor='k', label=filename.split('.')[0])

    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf.crs.to_string())

    # Add labels and title
    ax.legend(title='Shapefiles')
    ax.set_title("Shapefiles Map")

    # Show the plot
    plt.show()

# Specify your folder path here
folder_path = '../Japan_Data/gm-jpn-all_u_2_2'
plot_shapefiles(folder_path)
