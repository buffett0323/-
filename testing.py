import os
import torch
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.ops import unary_union
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from test_visualization import visualize_admin_result, visualize_cut_result, model_comp

path = 'data/'

# Testing the model result
def testing_cut(model, criterion, test_data, test_labels, plotting=False):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        test_loss = criterion(outputs, test_labels)
        print(f'The testing loss is {test_loss.item()}.\n')

        pred = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
        print(f'The accuracy is {round(accuracy*100, 2)} %\n')
        
        visualize_cut_result(test_labels.cpu(), pred.cpu())

    return round(accuracy*100, 2)




# Testing the hybrid model result
def testing_cut_hybrid(model, criterion, test_seq, test_static, test_labels, plotting=False):
    model.eval()
    with torch.no_grad():
        outputs = model(test_seq, test_static)
        test_loss = criterion(outputs, test_labels.flatten())
        print(f'The testing loss is {test_loss.item()}.\n')

        pred = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
        print(f'The accuracy is {round(accuracy*100, 2)} %\n')
        
        visualize_cut_result(test_labels.cpu(), pred.cpu())

    return round(accuracy*100, 2)




# Testing for the y2 level use
def testing(model, criterion, test_data, test_labels, clust_cnt):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        test_loss = criterion(outputs, test_labels)
        print(f'The testing loss is {test_loss.item()}.\n')

        pred = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
        print(f'The accuracy is {round(accuracy*100, 2)} %\n')
        
        visualize_admin_result(test_labels.cpu(), pred.cpu(), clust_cnt=clust_cnt)

    return round(accuracy*100, 2)



# Testing the hybrid model result
def testing_hybrid(model, criterion, test_seq, test_static, test_data, test_labels, x_scaler, clust_cnt, md_param=''):
    model.eval()
    with torch.no_grad():
        outputs = model(test_seq, test_static)
        test_loss = criterion(outputs, test_labels.flatten())
        print(f'The testing loss is {test_loss.item()}.\n')

        pred = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
        print(f'The accuracy is {round(accuracy*100, 2)} %\n')
        
        get_hybrid_adj_acc(test_labels.cpu(), pred.cpu())
        visualize_admin_result(test_labels.cpu(), pred.cpu(), clust_cnt, md_param=md_param)
        model_comp(model, test_seq, test_static, test_data, test_labels, x_scaler)

    return round(accuracy*100, 2)



# Testing for the adjacency model result
def get_hybrid_adj_acc(actual, pred):
        
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
    
    """ Creating Adjacency Matrix """
    adj_mat_np = np.zeros((len(gdf2), len(gdf2)), dtype=int)
    for i, poly1 in enumerate(gdf2.geometry):
        adj_mat_np[i, i] = 1
        for j, poly2 in enumerate(gdf2.geometry[i+1:], start=i+1): # Start from i+1 to avoid self-comparison and repeat comparisons
            if poly1.touches(poly2):
                adj_mat_np[i, j] = 1
                adj_mat_np[j, i] = 1 

    adj_mat_df = pd.DataFrame(adj_mat_np, index=gdf2.y_id, columns=gdf2.y_id)
    
    # Get the correct count by using adjacency matrix
    correct_cnt = 0
    for act, pr in zip(actual.tolist(), pred.tolist()):
        if adj_mat_df.loc[act, pr] == 1:
            correct_cnt += 1
    print(f"The adjacency accuracy is {round(100 * correct_cnt/len(pred), 2)} %\n")
    
