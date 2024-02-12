import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from params import SEQ_LENGTH, occupation, trip_purpose
plotting = False


seq_data = np.load(f'data/seqRegion_{SEQ_LENGTH}.npy', allow_pickle=True)
seq_2d = seq_data.reshape(seq_data.shape[0] * seq_data.shape[1], seq_data.shape[2])

""" 1. Splitting data by gender, which is the second column """
seq_2d_g1 = seq_2d[seq_2d[:, 1] == 1]
seq_2d_g2 = seq_2d[seq_2d[:, 1] == 2]

# Remove the gender column
seq_2d_g1 = np.delete(seq_2d_g1, 1, axis=1)
seq_2d_g2 = np.delete(seq_2d_g2, 1, axis=1)

seq_3d_g1 = seq_2d_g1.reshape((int(seq_2d_g1.shape[0] / SEQ_LENGTH), SEQ_LENGTH, seq_2d_g1.shape[1]))
seq_3d_g2 = seq_2d_g2.reshape((int(seq_2d_g2.shape[0] / SEQ_LENGTH), SEQ_LENGTH, seq_2d_g2.shape[1]))

np.save(f'data/seqRegion_{SEQ_LENGTH}_G1.npy', seq_3d_g1)
np.save(f'data/seqRegion_{SEQ_LENGTH}_G2.npy', seq_3d_g2)

""" 2. Splitting data by age, which is the third column """
age = seq_2d[:, 2].tolist()
age_list, age_name = [], []
for a in set(age):
    st = f'{a*5}-{(a+1)*5}'
    age_name.append(st)
    age_list.append(age.count(a))

# plotting
if plotting:
    plt.figure(figsize=(12,9))
    plt.plot(age_list)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(range(len(age_list)), age_name)
    for i in range(len(age_list)):
        plt.text(i, age_list[i], str(age_list[i]), ha='center', va='bottom')
    plt.title('The distribution of ages')
    plt.savefig('visualization/age_distribution.png')
    plt.show()

# Split by ages
mask_group_1, mask_group_2, mask_group_3 = [], [], []
for a in seq_data[:, -1, 2]:
    if a < 4: # 0-20 (grp 0-3)
        mask_group_1.append(True)
        mask_group_2.append(False)
        mask_group_3.append(False)
    elif a >= 4 and a <= 12: # 20-65 (grp 4-12)
        mask_group_1.append(False)
        mask_group_2.append(True)
        mask_group_3.append(False)
    else: # 65 up (grp 13~)
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(True)
        
        
# Split the array
data_group_1 = seq_data[mask_group_1]
data_group_2 = seq_data[mask_group_2]
data_group_3 = seq_data[mask_group_3]

np.save(f'data/seqRegion_{SEQ_LENGTH}_A1.npy', data_group_1) # 0-20
np.save(f'data/seqRegion_{SEQ_LENGTH}_A2.npy', data_group_2) # 20-65
np.save(f'data/seqRegion_{SEQ_LENGTH}_A3.npy', data_group_3) # 65up
print("Successfully split the data into 3 groups!", 
      f"The shapes are {data_group_1.shape}; {data_group_2.shape}; {data_group_3.shape}")


""" 3. Splitting data by time zone """
hr_list = []
mask_group_1, mask_group_2, mask_group_3 = [], [], []
for row in seq_data[:, :, 0]:
    datetime_row = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in row]
    hr_row = [d.hour for d in datetime_row]
    hr_list.append(hr_row[-1])
    
    if hr_row[-1] <= 8: # 0-9
        mask_group_1.append(True)
        mask_group_2.append(False)
        mask_group_3.append(False)
    elif hr_row[-1] > 8 and hr_row[-1] <= 16: # 9-17
        mask_group_1.append(False)
        mask_group_2.append(True)
        mask_group_3.append(False)
    else: # 17-24
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(True)


# Split the array
data_group_1 = seq_data[mask_group_1]
data_group_2 = seq_data[mask_group_2]
data_group_3 = seq_data[mask_group_3]

np.save(f'data/seqRegion_{SEQ_LENGTH}_T1.npy', data_group_1) # 0-9
np.save(f'data/seqRegion_{SEQ_LENGTH}_T2.npy', data_group_2) # 9-17
np.save(f'data/seqRegion_{SEQ_LENGTH}_T3.npy', data_group_3) # 17-24
print("Successfully split the data into 3 groups!", 
      f"The shapes are {data_group_1.shape}; {data_group_2.shape}; {data_group_3.shape}")

# plotting
if plotting:
    hr_count = [hr_list.count(i) for i in range(24)]
    plt.figure(figsize=(12,9))
    plt.plot(hr_count)
    plt.xlabel('hour')
    plt.ylabel('Count')
    for i in range(len(hr_count)):
        plt.text(i, hr_count[i], str(hr_count[i]), ha='center', va='bottom')
    plt.title('The distribution of hour')
    plt.savefig('visualization/hr_distribution.png')
    plt.show()


""" 4. Work type """
def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None 

work_list = [seq_data[i, 0, 3] for i in range(seq_data.shape[0])]
work_name = [occupation[w] for w in set(work_list)]
work_count = [work_list.count(w) for w in set(work_list)]
sorted_indices = sorted(range(len(work_count)), key=work_count.__getitem__, reverse=True)

# Use the sorted indices to sort both lists
sorted_work_count = [work_count[i] for i in sorted_indices]
sorted_work_name = [work_name[i] for i in sorted_indices]
sorted_work_key = [get_key_from_value(occupation, wn) for wn in sorted_work_name]

mask_group_1, mask_group_2, mask_group_3, mask_group_4 = [], [], [], []
for work in seq_data[:, 0, 3]:
    if work == sorted_work_key[0]:
        mask_group_1.append(True)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(False)
    elif work == sorted_work_key[1]:
        mask_group_1.append(False)
        mask_group_2.append(True)
        mask_group_3.append(False)
        mask_group_4.append(False)
    elif work == sorted_work_key[2]:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(True)
        mask_group_4.append(False)
    elif work == sorted_work_key[3]:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(True)
    else:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(False)

# Split the array
data_group_1 = seq_data[mask_group_1]
data_group_2 = seq_data[mask_group_2]
data_group_3 = seq_data[mask_group_3]
data_group_4 = seq_data[mask_group_4]
    
np.save(f'data/seqRegion_{SEQ_LENGTH}_W1.npy', data_group_1) # Group1
np.save(f'data/seqRegion_{SEQ_LENGTH}_W2.npy', data_group_2) # Group2
np.save(f'data/seqRegion_{SEQ_LENGTH}_W3.npy', data_group_3) # Group3
np.save(f'data/seqRegion_{SEQ_LENGTH}_W4.npy', data_group_4) # Group4
print("Successfully split the work data into 4 groups!", 
      f"The shapes are {data_group_1.shape}; {data_group_2.shape}; {data_group_3.shape}; {data_group_4.shape}")
print(sorted_work_key[0:4])

""" 5. Last time's Trip Purpose """
tp_list = [seq_data[i, -1, 4] for i in range(seq_data.shape[0])]
tp_name = [trip_purpose[w] for w in set(tp_list)]
tp_count = [tp_list.count(w) for w in set(tp_list)]
sorted_indices = sorted(range(len(tp_count)), key=tp_count.__getitem__, reverse=True)

# Use the sorted indices to sort both lists
sorted_tp_name = [tp_name[i] for i in sorted_indices]
sorted_tp_count = [tp_count[i] for i in sorted_indices]
sorted_tp_key = [get_key_from_value(trip_purpose, wn) for wn in sorted_tp_name]

mask_group_1, mask_group_2, mask_group_3, mask_group_4 = [], [], [], []
for tp in seq_data[:, -1, 4]:
    if tp == sorted_tp_key[0]:
        mask_group_1.append(True)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(False)
    elif tp == sorted_tp_key[1]:
        mask_group_1.append(False)
        mask_group_2.append(True)
        mask_group_3.append(False)
        mask_group_4.append(False)
    elif tp == sorted_tp_key[2]:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(True)
        mask_group_4.append(False)
    elif tp == sorted_tp_key[3]:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(True)
    else:
        mask_group_1.append(False)
        mask_group_2.append(False)
        mask_group_3.append(False)
        mask_group_4.append(False)
        

# Split the array
data_group_1 = seq_data[mask_group_1]
data_group_2 = seq_data[mask_group_2]
data_group_3 = seq_data[mask_group_3]
data_group_4 = seq_data[mask_group_4]
    
np.save(f'data/seqRegion_{SEQ_LENGTH}_TP{sorted_tp_key[0]}_0.npy', data_group_1) # Group1: 99
np.save(f'data/seqRegion_{SEQ_LENGTH}_TP{sorted_tp_key[1]}_1.npy', data_group_2) # Group2: 3
np.save(f'data/seqRegion_{SEQ_LENGTH}_TP{sorted_tp_key[2]}_2.npy', data_group_3) # Group3: 4
np.save(f'data/seqRegion_{SEQ_LENGTH}_TP{sorted_tp_key[3]}_3.npy', data_group_4) # Group4: 1
print("Successfully split the trip purpose data into 4 groups!", 
      f"The shapes are {data_group_1.shape}; {data_group_2.shape}; {data_group_3.shape}; {data_group_4.shape}")
print(sorted_tp_key[0:4])

""" 6. Work Group and Time slots """
group_id = []
mask_6 = [0 for _ in range(seq_data.shape[0])]
for i in range(seq_data.shape[0]):
    dt = datetime.strptime(seq_data[i, -1, 0], '%Y-%m-%d %H:%M:%S')
    if dt.hour > 8 and dt.hour <= 16 and seq_data[i, 0, 3] == sorted_work_key[0]: 
        mask_6[i] = 1
    elif dt.hour > 8 and dt.hour <= 16 and seq_data[i, 0, 3] == sorted_work_key[1]:
        mask_6[i] = 2
    elif dt.hour > 8 and dt.hour <= 16 and seq_data[i, 0, 3] == sorted_work_key[2]:
        mask_6[i] = 3
    elif dt.hour > 16 and seq_data[i, 0, 3] == sorted_work_key[0]:
        mask_6[i] = 4
    elif dt.hour > 16 and seq_data[i, 0, 3] == sorted_work_key[1]:
        mask_6[i] = 5
    elif dt.hour > 16 and seq_data[i, 0, 3] == sorted_work_key[2]:
        mask_6[i] = 6


data_group_1 = seq_data[[True if i == 1 else False for i in mask_6]]
data_group_2 = seq_data[[True if i == 2 else False for i in mask_6]]
data_group_3 = seq_data[[True if i == 3 else False for i in mask_6]]
data_group_4 = seq_data[[True if i == 4 else False for i in mask_6]]
data_group_5 = seq_data[[True if i == 5 else False for i in mask_6]]
data_group_6 = seq_data[[True if i == 6 else False for i in mask_6]]
    
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_1.npy', data_group_1)
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_2.npy', data_group_2)
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_3.npy', data_group_3)
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_4.npy', data_group_4)
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_5.npy', data_group_5)
np.save(f'data/seqRegion_{SEQ_LENGTH}_Mixed_6.npy', data_group_6)

print("Successfully split the data into 6 groups!", 
      f"The shapes are {data_group_1.shape}; {data_group_2.shape}; {data_group_3.shape}; {data_group_4.shape}; {data_group_5.shape}; {data_group_6.shape}")
