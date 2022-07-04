#%%
import numpy as np
import os
from os.path import join as opj

path = 'C:/Users/JJY/Desktop/coco/labels/train2014'
label_list = os.listdir(path)
print(len(label_list))

iter = 0
for label_path in label_list: 
    boxes = np.loadtxt(opj(path, label_path)).reshape(-1, 5)
    if boxes.shape[1] != 5:
        print(label_path, boxes.shape[1])
    else:
        if boxes.shape[0] == 2:
            print(label_path, boxes.shape[0])
            iter += 1
            if iter == 100:
                break

#%%
import numpy as np
import os
from os.path import join as opj

path = 'C:/Users/JJY/Desktop/coco/labels/train2014'
label_list = os.listdir(path)
print(len(label_list))

iter = 0
for label_path in label_list: 
    boxes = np.loadtxt(opj(path, label_path)).reshape(-1, 5)
    iter += 1
    if np.array([0]) in boxes[:, 0]:
        print(label_path, 'has class 0')
    if iter == 1000:
        break
# %%
