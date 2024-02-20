import os
import folium
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from fractions import Fraction
from shapely.geometry import Polygon
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from params import SEQ_LENGTH, CUT, trip_purpose, transport_type, landuse, age_hierarchy
path = 'data'


# Function to generate popup content
def generate_popup(row):
    return f'Accuracy: {row["acc_float"]}%'


# Define a function to style the polygons
def style_function(feature):
    return {
        'fillColor': '#ffff00',  # or some logic to choose color
        'color': 'black',
        'weight': 1,
        'dashArray': '5, 5'
    }


def style_function2(feature, actual_target, pred):
    if feature['properties']['y_id'] == actual_target:
        return {'fillColor': 'green', 'color': 'green'}
    elif feature['properties']['y_id'] == pred:
        return {'fillColor': 'red', 'color': 'red'}
    else:
        return {'fillColor': 'lightblue', 'color': 'blue'}


def visualize_cut_result(test_labels, pred):
    cor_cnt, ttl_cnt = [0 for _ in range(CUT*CUT)], [0 for _ in range(CUT*CUT)]
    for l, p in zip(test_labels, pred):
        if l == p:
            cor_cnt[l] += 1
        ttl_cnt[l] += 1

    acc_100 = [round(i*100/j, 0) if j != 0 else 0 for i, j in zip(cor_cnt, ttl_cnt)]
    acc_frac = [Fraction(i, j) if j != 0 else 0 for i, j in zip(cor_cnt, ttl_cnt)]


    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}.npy', allow_pickle=True) # (6, 770400, 8)
    seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
    max_lon, min_lon = max(seq_2d[:,6].tolist()), min(seq_2d[:,6].tolist())
    max_lat, min_lat = max(seq_2d[:,7].tolist()), min(seq_2d[:,7].tolist())

    y_pd = pd.DataFrame({
        'y_id': [i for i in range(CUT*CUT)],
        'tl_count': ttl_cnt,
        'acc_count': cor_cnt,
        'acc_float': acc_100,
        'acc_frac': acc_frac
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

    # Plotting
    fig, ax1 = plt.subplots()
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')

    gdf2.boundary.plot(ax=ax1, color='lightblue', label='Level 2')
    gdf1.boundary.plot(ax=ax1, color='black', label='Level 1', alpha=0.2)
    gdf.plot(column='acc_float', ax=ax1, legend=True, cmap='OrRd')


    for i in range(1, CUT):
        x_tmp = min_lon + i * (max_lon - min_lon) / CUT
        y_tmp = min_lat + i * (max_lat - min_lat) / CUT
        ax1.plot([x_tmp, x_tmp], [min_lat, max_lat], color='gray', linestyle='--', alpha=0.2)
        ax1.plot([min_lon, max_lon], [y_tmp, y_tmp], color='gray', linestyle='--', alpha=0.2)

    ax1.plot([min_lon, min_lon], [min_lat, max_lat], color='gray', linestyle='-')
    ax1.plot([max_lon, max_lon], [min_lat, max_lat], color='gray', linestyle='-')
    ax1.plot([min_lon, max_lon], [min_lat, min_lat], color='gray', linestyle='-')
    ax1.plot([min_lon, max_lon], [max_lat, max_lat], color='gray', linestyle='-')

    # # Annotate the plot
    # for idx, row in gdf.iterrows():
    #     centroid = row.geometry.centroid # Get the centroid of the polygon
    #     plt.annotate(text=row['acc_frac'], xy=(centroid.x, centroid.y), xytext=(3,3), ha='center', va='center', fontsize=6, textcoords="offset points")

    longitude_range = [round(min_lon, 1) - 0.3 , round(max_lon, 1) + 0.3]
    latitude_range = [round(min_lat, 1) - 0.3, round(max_lat, 1) + 0.3]
    ax1.set_xlim(longitude_range)
    ax1.set_ylim(latitude_range)
    ax1.legend(loc='upper right')


    plt.title(f'Japan District with Cut {CUT} Test Accuracy')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'visualization/visual_{CUT}_test_acc.jpg')
    plt.show()

    # # Use foliumto draw
    # m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=10)
    # gdf_json = gdf.to_json()
    
    # # Add polygons to the map with color based on 'acc_float' and tooltip for each polygon
    # folium.Choropleth(
    #     geo_data=gdf_json,
    #     data=gdf,
    #     columns=['poly_id', 'acc_float'],
    #     key_on='feature.properties.poly_id',
    #     fill_color='YlOrRd',  # Color scale
    #     fill_opacity=0.7,
    #     line_opacity=0.2,
    #     legend_name='Accuracy (%)'
    # ).add_to(m)

    # # Alternatively, use GeoJson for more customization
    # for _, row in gdf.iterrows():
    #     tooltip = generate_popup(row)
    #     folium.GeoJson(
    #         row['geometry'],
    #         tooltip=tooltip,
    #         style_function=lambda x: {'fillColor': '#ffaf00', 'color': '#00af00', 'weight': 1}
    #     ).add_to(m)

    # # Save the map to an HTML file
    # m.save(f'visualization/visual_{CUT}_test_acc.html')

    

def visualize_admin_result(test_labels, pred, clust_cnt, md_param=''):
    # Deal with the mapping of y labels
    y_init, y_label = [], []
    with open(os.path.join(path, 'y_mapping.txt'), 'r') as f:
        for line in f:
            y_init.append(int(line.split(':')[0]))
            gid_num = int(line.split(':')[-1].split('.')[0])
            gid_num1, gid_num2 = int(gid_num//100), gid_num%100
            y_label.append(f'JPN.{gid_num1}.{gid_num2}_1')

    cor_cnt, ttl_cnt = [0 for _ in range(clust_cnt)], [0 for _ in range(clust_cnt)]
    for l, p in zip(test_labels, pred):
        if l == p:
            cor_cnt[l] += 1
        ttl_cnt[l] += 1

    acc_100 = [round(i*100/j, 1) if j != 0 else 0 for i, j in zip(cor_cnt, ttl_cnt)]
    acc_frac = [str(Fraction(i, j)) if j != 0 else 0 for i, j in zip(cor_cnt, ttl_cnt)]

    seq_data = np.load(f'{path}/newDataShift_{SEQ_LENGTH}.npy', allow_pickle=True) # (6, 770400, 8)
    seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])
    max_lon, min_lon = max(seq_2d[:,6].tolist()), min(seq_2d[:,6].tolist())
    max_lat, min_lat = max(seq_2d[:,7].tolist()), min(seq_2d[:,7].tolist())
    
    y_pd = pd.DataFrame({
        'y_id': y_init, #[i for i in range(clust_cnt)],
        'y_map_label': y_label, # JPN.1.7_1
        'tl_count': ttl_cnt,
        'acc_count': cor_cnt,
        'acc_float': acc_100,
        'acc_frac': acc_frac
    })
    
    # Plotting
    fig, ax = plt.subplots()
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    gdf1 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_1.shp')
 
    # Filter
    gdf2 = gdf2[gdf2["GID_2"].isin(y_pd["y_map_label"])]
    gdf2 = gdf2.merge(y_pd, left_on='GID_2', right_on='y_map_label')

    gdf2.boundary.plot(ax=ax, color='lightblue', label='Level 2', alpha=0.8, linewidth=0.2)
    gdf1.boundary.plot(ax=ax, color='black', label='Level 1', alpha=0.2, linewidth=0.5)
    gdf2.plot(column='acc_float', ax=ax, legend=True, cmap='OrRd')


    # # Annotate the plot
    # for idx, row in gdf2.iterrows():
    #     centroid = row.geometry.centroid # Get the centroid of the polygon
    #     plt.annotate(text=row['acc_frac'], xy=(centroid.x, centroid.y), xytext=(3,3), ha='center', va='center', fontsize=4, textcoords="offset points")

    longitude_range = [138.7, 140.5]
    latitude_range = [34.8, 36.5]
    ax.set_xlim(longitude_range)
    ax.set_ylim(latitude_range)
    ax.tick_params(axis='x', labelsize=7)  
    ax.tick_params(axis='y', labelsize=7)
    ax.legend()

    accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
    plt.title(f'Japan Level 2 Prediction: Test Accuracy = {round(accuracy*100, 2)}%')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'{path}/visual_lv2_test_seq_{SEQ_LENGTH}_{md_param}.png', dpi=300)
    # plt.show()
    
    
    # Create a Folium map & Adjust the location and zoom_start as needed
    gdf2_json = gdf2.to_json()
    average_longitude = (min_lon + max_lon) / 2
    average_latitude = (min_lat + max_lat) / 2
    m = folium.Map(location=[average_latitude, average_longitude], zoom_start=5)

    # Add Choropleth layer
    ch_layer = folium.Choropleth(
        geo_data=gdf2_json,
        name='Choropleth',
        data=gdf2,
        columns=['GID_2', 'acc_float'],  # Adjust these column names based on your DataFrame
        key_on='feature.properties.GID_2',  # This should match the GeoJSON property name
        fill_color='YlOrRd',  # Choose a color palette
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Accuracy'
    )
    ch_layer.add_to(m, name='Choropleth Layer')

    # Create a GeoJson layer with a tooltip
    geojson_data = json.loads(gdf2_json)
    geojson_layer = folium.GeoJson(
        geojson_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_2', 'acc_float', 'acc_frac'],  # Fields from your GeoJSON properties
            aliases=['Name', 'Accuracy', 'Proportion'],  # Names you want to show in the tooltip
            localize=True,
            sticky=False
        )
    )
    geojson_layer.add_to(m, name='Accuracy View Layer')

    # Add layer control and save the map
    folium.LayerControl().add_to(m)
    m.save(f'{path}/visual_lv2_test_seq_{SEQ_LENGTH}_{md_param}.html')
    
    
    
def model_comp(model, test_seq, test_static, test_data, test_labels, x_scaler):
    LOOKBACK = SEQ_LENGTH - 1
    test_labels = test_labels.cpu().tolist()
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_seq, test_static)
        pred = torch.argmax(outputs, dim=1)

    # Filter the wrong data
    wrong_mask = [True if i != j else False for i, j in zip(test_labels, pred)]
    right_mask = [True if i == j else False for i, j in zip(test_labels, pred)]
    
    # Get the original data
    test_2d = test_data.reshape((test_data.shape[0] * test_data.shape[1], test_data.shape[2]))
    test_2d_orig = x_scaler.inverse_transform(test_2d)
 
    columns_to_round = [i for i in range(test_2d_orig.shape[1]) if i not in [6, 7]]
    test_2d_orig[:, columns_to_round] = np.round(test_2d_orig[:, columns_to_round]).astype(int)
    test_data_orig = test_2d_orig.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    test_data_wrong, test_data_right = test_data_orig[wrong_mask], test_data_orig[right_mask]
  
    # Get the base map
    gdf2 = gpd.read_file('../Japan_Data/gadm41_JPN_shp/gadm41_JPN_2.shp')
    y_init, y_label = [], []
    with open(os.path.join(path, 'y_mapping.txt'), 'r') as f:
        for line in f:
            y_init.append(int(line.split(':')[0]))
            gid_num = int(line.split(':')[-1].split('.')[0])
            gid_num1, gid_num2 = int(gid_num//100), gid_num%100
            y_label.append(f'JPN.{gid_num1}.{gid_num2}_1')
          
    y_pd = pd.DataFrame({'y_id': y_init, 'y_map_label': y_label})
    gdf2 = gdf2[gdf2["GID_2"].isin(y_label)]
    gdf2 = gdf2.merge(y_pd, left_on='GID_2', right_on='y_map_label')
    
    """ Creating Adjacency Matrix """
    adj_mat_np = np.zeros((len(gdf2), len(gdf2)), dtype=int)
    for i, poly1 in enumerate(gdf2.geometry):
        adj_mat_np[i, i] = 1
        for j, poly2 in enumerate(gdf2.geometry[i+1:], start=i+1): # Start from i+1 to avoid self-comparison and repeat comparisons
            if poly1.touches(poly2):
                adj_mat_np[i, j] = 1
                adj_mat_np[j, i] = 1 

    adj_mat_df = pd.DataFrame(adj_mat_np, index=gdf2.y_id, columns=gdf2.y_id)
    
    # Get the wrongly predicted adjacency mask
    adj_wrong_mask = [False if adj_mat_df.loc[int(act), int(pr)] == 1 else True for act, pr in zip(test_labels, pred)]
    y_wrong = [val for val, mask in zip(test_labels, adj_wrong_mask) if mask]
    pred_wrong = [val for val, mask in zip(pred, adj_wrong_mask) if mask]    



    """ Statistics for wrongly prediction only for seq_length=4 """
    # tp_dict = dict()
    # for i in trip_purpose.values():
    #     tp_dict[i] = 0
        
    # for i in range(test_data_orig.shape[0]):
    #     if adj_wrong_mask[i]:
    #         tp_dict[trip_purpose[test_data_orig[i, -1, 4]]] += 1
    
    # print(tp_dict)
    
    
    # Use folium to create the map
    gdf2_json = json.loads(gdf2.to_json())
    m = folium.Map(location=[35.5, 139.75], zoom_start=9,  tiles='CartoDB positron')
    
    coordinates_sets, y_actual = [], []
    for i in range(10):
        tmp_lon = test_data_wrong[i, :LOOKBACK, 6].tolist()
        tmp_lat = test_data_wrong[i, :LOOKBACK, 7].tolist()
        trip_p = [trip_purpose[int(j)] for j in test_data_wrong[i, :LOOKBACK, 4]]
        trans = [transport_type[int(j)] for j in test_data_wrong[i, :LOOKBACK, 5]] 
        info_list = [f'Time:{idx+1}; Trip_Purpose:{trip_p[idx]}; Transport_Type:{trans[idx]}' for idx in range(LOOKBACK)]
        coordinates_sets.append((l1, l2, info) for l1, l2, info in zip(tmp_lon, tmp_lat, info_list))
        y_actual.append(y_wrong[i])
        

    # Add each set of coordinates and corresponding polygon as a separate layer
    for idx, coord_set in enumerate(coordinates_sets):
        layer = folium.FeatureGroup(name=f'Layer {idx}', show=idx==0)

        for i, (lon, lat, info) in enumerate(coord_set):
            folium.Marker(
                location=[lat, lon],
                tooltip=info,
                icon=folium.Icon(color='red')# icon=folium.DivIcon(html=f'<div style="font-size: 10pt">{i}</div>'),
            ).add_to(layer)
            
        folium.GeoJson(
            gdf2_json,
            style_function=lambda feature, idx=idx: style_function2(feature, y_actual[idx], pred_wrong[idx])
        ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl().add_to(m)
    m.save('visualization/y_level2.html')



    """ 1. Do the data analysis for trip purpose """
    acc_tp = dict()
    for tp, tp_val in trip_purpose.items():
        # if tp != 99:
        mask = [True if i == tp else False for i in test_data_orig[:, -1, 4].tolist()]
        test_seq1 = test_seq[mask]
        test_static1 = test_static[mask]
        test_labels1 = [val for val, mask in zip(test_labels, mask) if mask]
        
        model.eval()
        with torch.no_grad():
            outputs1 = model(test_seq1, test_static1)
            pred1 = torch.argmax(outputs1, dim=1).cpu().tolist()
            # accuracy1 = accuracy_score(test_labels1, pred1)
            # acc_tp[tp_val] = round(accuracy1 * 100, 1)
            
            # Get the correct count by using adjacency matrix
            correct_cnt = 0
            for act, pr in zip(test_labels1, pred1):
                if adj_mat_df.loc[act, pr] == 1:
                    correct_cnt += 1
            acc_tp[tp_val] = round(100 * correct_cnt/len(pred1), 2)
            

    sorted_accuracy = sorted(acc_tp.items(), key=lambda x: x[1], reverse=True)
    labels, accuracies = zip(*sorted_accuracy)

    # Create the Plotly bar chart
    fig = go.Figure(data=[go.Bar(x=labels, y=accuracies, text=accuracies, textposition='auto', marker_color='lightblue')])
    fig.update_layout(
        title='Different Trip Purpose', 
        xaxis_title='Trip Purpose', 
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0,100]),
        template='plotly_white' 
    )
    
    # Save the figure to a file
    file_path = 'plotly_result/hybrid_lstm_trip_purpose.png'
    pio.write_image(fig, file_path)
    
    """ 2. Do the data analysis for transport type """
    acc_tt = dict()
    for tt, tt_val in transport_type.items():
        if tt != 99: # tt != 15 and tt != 97 and
            mask = [True if i == tt else False for i in test_data_orig[:, -1, 5].tolist()]
            if mask.count(True) == 0: continue
            test_seq1 = test_seq[mask]
            test_static1 = test_static[mask]
            test_labels1 = [val for val, mask in zip(test_labels, mask) if mask]
                
            model.eval()
            with torch.no_grad():
                outputs1 = model(test_seq1, test_static1)
                pred1 = torch.argmax(outputs1, dim=1).cpu().tolist()
                # accuracy1 = accuracy_score(test_labels1, pred1)
                # acc_tt[tt_val] = round(accuracy1 * 100, 1)
                
                # Get the correct count by using adjacency matrix
                correct_cnt = 0
                for act, pr in zip(test_labels1, pred1):
                    if adj_mat_df.loc[act, pr] == 1:
                        correct_cnt += 1
                acc_tt[tt_val] = round(100 * correct_cnt/len(pred1), 2)

    sorted_accuracy = sorted(acc_tt.items(), key=lambda x: x[1], reverse=True)
    labels, accuracies = zip(*sorted_accuracy)

    # Create the Plotly bar chart
    fig = go.Figure(data=[go.Bar(x=labels, y=accuracies, text=accuracies, textposition='auto', marker_color='lightblue')])
    fig.update_layout(
        title='Different Transport Type', 
        xaxis_title='Transport Type', 
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0,100]),
        template='plotly_white' 
    )
    
    # Save the figure to a file
    file_path = 'plotly_result/hybrid_lstm_transport_type.png'
    pio.write_image(fig, file_path)
    
    
    
    """ 3. Do the data analysis on the last TIME HOUR """
    acc_tm = dict()
    time_interval = {
        "8-10": [8,10],
        "10-16": [10,16],
        "16-19": [16,19],
        "19-24": [19,24]
    }
    for tm, tm_val in time_interval.items():
        mask = [True if i >= tm_val[0] and i < tm_val[1] else False for i in test_data_orig[:, -1, -1].tolist()]
        if mask.count(True) == 0: continue
        test_seq1 = test_seq[mask]
        test_static1 = test_static[mask]
        test_labels1 = [val for val, mask in zip(test_labels, mask) if mask]
        
        model.eval()
        with torch.no_grad():
            outputs1 = model(test_seq1, test_static1)
            pred1 = torch.argmax(outputs1, dim=1).cpu().tolist()
            # accuracy1 = accuracy_score(test_labels1, pred1)
            # acc_tm[tm] = round(accuracy1 * 100, 1)
            
            # Get the correct count by using adjacency matrix
            correct_cnt = 0
            for act, pr in zip(test_labels1, pred1):
                if adj_mat_df.loc[act, pr] == 1:
                    correct_cnt += 1
            acc_tm[tm] = round(100 * correct_cnt/len(pred1), 2)

    # Create the Plotly bar chart
    fig = go.Figure(data=[go.Bar(
        x=list(acc_tm.keys()),
        y=list(acc_tm.values()),
        text=[str(val) for val in acc_tm.values()],
        textposition='auto', marker_color='lightblue')])
    fig.update_layout(
        title='Last Time Position', 
        xaxis_title='Time zone', 
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0,100]),
        template='plotly_white' 
    )
    
    # Save the figure to a file
    file_path = 'plotly_result/hybrid_lstm_time_position.png'
    pio.write_image(fig, file_path)
    
    
    """ 4. Do the data analysis on the last TIME INTERVAL """
    acc_tm = dict()
    time_interval = {
        "0~20": [0, 20],
        "20~40": [20, 40],
        "40~60": [40, 60],
        "60~80": [60, 80],
        "80~100": [80, 100],
        "100~120": [100, 120],
        "120~140": [120, 140],
        "140~160": [140, 160],
        "160~180": [160, 180],
        "180~": [180, 1e5]
    }
    for tm, tm_val in time_interval.items():
        mask = [True if i >= tm_val[0] and i < tm_val[1] else False for i in test_data_orig[:, -1, 0].tolist()]
        if mask.count(True) == 0: continue
        test_seq1 = test_seq[mask]
        test_static1 = test_static[mask]
        test_labels1 = [val for val, mask in zip(test_labels, mask) if mask]
        
        model.eval()
        with torch.no_grad():
            outputs1 = model(test_seq1, test_static1)
            pred1 = torch.argmax(outputs1, dim=1).cpu().tolist()
            # accuracy1 = accuracy_score(test_labels1, pred1)
            # acc_tm[tm] = round(accuracy1 * 100, 1)
            
            # Get the correct count by using adjacency matrix
            correct_cnt = 0
            for act, pr in zip(test_labels1, pred1):
                if adj_mat_df.loc[act, pr] == 1:
                    correct_cnt += 1
            acc_tm[tm] = round(100 * correct_cnt/len(pred1), 2)

    # Create the Plotly bar chart
    fig = go.Figure(data=[go.Bar(
        x=list(acc_tm.keys()),
        y=list(acc_tm.values()),
        text=[str(val) for val in acc_tm.values()],
        textposition='auto', marker_color='lightblue')])
    fig.update_layout(
        title='Different Time Interval', 
        xaxis_title='Time zone', 
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0,100]),
        template='plotly_white' 
    )
    
    # Save the figure to a file
    file_path = 'plotly_result/hybrid_lstm_time_interval.png'
    pio.write_image(fig, file_path) 
    
    
    """ 5. Do the data analysis on different AGE """
    acc_age = dict()
    for ag, ag_val in age_hierarchy.items():
        
        mask = [True if i == ag else False for i in test_data_orig[:, -1, 2].tolist()]
        if mask.count(True) <= 10: continue
        
        test_seq1 = test_seq[mask]
        test_static1 = test_static[mask]
        test_labels1 = [val for val, mask in zip(test_labels, mask) if mask]
        
        model.eval()
        with torch.no_grad():
            outputs1 = model(test_seq1, test_static1)
            pred1 = torch.argmax(outputs1, dim=1).cpu().tolist()
            # accuracy1 = accuracy_score(test_labels1, pred1)
            # acc_age[ag_val] = round(accuracy1 * 100, 1)

            # Get the correct count by using adjacency matrix
            correct_cnt = 0
            for act, pr in zip(test_labels1, pred1):
                if adj_mat_df.loc[act, pr] == 1:
                    correct_cnt += 1
            acc_age[ag_val] = round(100 * correct_cnt/len(pred1), 2)

    # Create the Plotly bar chart
    fig = go.Figure(data=[go.Bar(
        x=list(acc_age.keys()),
        y=list(acc_age.values()),
        text=[str(val) for val in acc_age.values()],
        textposition='auto', marker_color='lightblue')])
    fig.update_layout(
        title='Different Age', 
        xaxis_title='Age Hierarchy', 
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0,100]),
        template='plotly_white' 
    )
    
    # Save the figure to a file
    file_path = 'plotly_result/hybrid_lstm_age.png'
    pio.write_image(fig, file_path)
    
    
    
    # """ 6. Use ML Methods to get the features """
    # X_seq = test_seq.reshape((test_seq.shape[0], test_seq.shape[1] * test_seq.shape[2]))
    # X_train = np.concatenate((test_static, X_seq), axis=1)
    # y_train = [1 if i == True else 0 for i in wrong_mask]

    
    # rf_model = RandomForestClassifier()
    # rf_model.fit(X_train, y_train)

    # importances = rf_model.feature_importances_
    # col = ["Gender", "Age", "Work"]
    # for i in range(LOOKBACK):
    #     for j in ["Time_diff", "Trip_purpose", "Transport", "Long", "Lat", "Landuse", "LandCover"]:
    #         col.append(f'{j}_{i+1}')
    
    # # Create a DataFrame for plotting
    # importance_df = pd.DataFrame({
    #     'Feature': col,
    #     'Importance': importances
    # })

    # # Sort the DataFrame by importance for better visualization
    # importance_df = importance_df.sort_values('Importance', ascending=False)

    # # Create the bar plot using Plotly
    # fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importances')
    # fig.update_layout(yaxis={'categoryorder':'total ascending'})
    # file_path = 'plotly_result/rf_importance_plotly.png'
    # pio.write_image(fig, file_path)
    
    
    
    # """ 7. Final analysis for the most importance part """
    # time_diff_last_wrong = test_data_wrong[:, LOOKBACK-1, 0].tolist()
    # time_diff_last_right = test_data_right[:, LOOKBACK-1, 0].tolist()

    # fig = go.Figure()
    # fig.add_trace(go.Box(y=time_diff_last_wrong, name='wrong'))
    # fig.add_trace(go.Box(y=time_diff_last_right, name='right'))

    # fig.update_layout(
    #     title='Box Plot of last time diff for wrong and right prediction',
    #     yaxis_title='Values',
    #     boxmode='group'
    # )

    # # Show the plot
    # # fig.show()
    # fig.write_html('plotly_result/last_time_diff.html')
    
    # file_path = 'plotly_result/last_time_diff.png'
    # pio.write_image(fig, file_path)

    



