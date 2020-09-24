import os
import h5py

import random
import numpy as np
import torch


def get_hdf5_data(file_path,labeled=False):
    with h5py.File(file_path,'r') as f:
        data=f['x'].value
        if labeled:
            label=f['y'].value
        else:
            label=[]
    return data,label

def split_data(data,label,split_ratio=0.1,seed_num=42):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split_ratio, random_state=seed_num)
    return X_train, X_test, y_train, y_test

def filter_label(data,label,select_label=0):
    select_idx=[label==select_label]
    print("Selected {} total {}".format(data[select_idx].shape[0],data.shape[0]))
    data=data[tuple(select_idx)]
    label=label[tuple(select_idx)]
    return data,label
    
def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
