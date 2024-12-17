import os
from params import SEQ_LENGTH
path = 'data/'


def label_mapping(label1, label2):
    old_label = [100*y1 + y2 for y1, y2 in zip(label1, label2)]
    new_label = []
    exist_labels = list(set(old_label))
    mapping = {exist_labels[i]: i for i in range(len(exist_labels))}

    # Store the mapping results
    with open(os.path.join(path, 'y_mapping.txt'), 'w') as f:
        for key, value in mapping.items():
            f.write(f"{value}:{key}\n")

    for i in range(len(old_label)):
        new_label.append(mapping[old_label[i]])
    if len(old_label) == len(new_label):
        return new_label
    return []


def label_removing(old_label):
    old_label, new_label = [int(i) for i in old_label], []
    exist_labels = list(set(old_label))
    mapping = {exist_labels[i]: i for i in range(len(exist_labels))} # 0:0, 3:3, 5:4, 7:5, ..., 233:213

    y_init, y_label = [], []
    with open(os.path.join(path, 'y_mapping.txt'), 'r') as f:
        for line in f:
            y_init.append(int(line.split(':')[0])) # 0-236
            y_label.append(float(line.split(':')[-1].split('\n')[0])) # GID
    
    new_mapping = {mapping[i]: j for i, j in zip(y_init, y_label) if i in exist_labels}
    with open(os.path.join(path, 'y_mapping.txt'), 'w') as f:
        for key, value in new_mapping.items():
            f.write(f"{key}:{value}\n")

    for i in range(len(old_label)):
        new_label.append(mapping[old_label[i]])
    if len(old_label) == len(new_label):
        return new_label
    return []

def region2id(y, cut):
    thres = float(1/cut)
    thres_range = [float(i * thres) for i in range(cut)]
    id = 0
    for i, thres in enumerate(thres_range):
        if y >= thres:
            id = i

    return id
