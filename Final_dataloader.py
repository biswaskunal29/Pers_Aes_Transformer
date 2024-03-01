import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import math
#import torch.nn as nn
#import torch.nn.functional as F
import csv
from Final_all_parts_blip_loader import get_all_parts_blip

traitmap = {'O':0 ,'C':1 ,'E':2 ,'A':3 ,'N':4}

def read_dataset_to_list(file_name):
    X_list = []
    y_list = []
    with open(file_name, newline='\n', encoding='utf-8') as csv_file:
        cf = csv.reader(csv_file, delimiter=',',quotechar='"')

        for row in cf:
            if(row[0]=='user_id'): continue
            X_list.append(row[0])
            y_list.append(traitmap[row[1]])
            
#            p_id = row[0]
#            class_id = traitmap[row[1]]        
#            ele = [p_id,class_id]
#            prof_list.append(ele)
    return X_list, y_list

class train_dataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
#        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
#        self.n_samples = xy.shape[0]
#
#        # here the first column is the class label, the rest are the features
#        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
#        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        
        X,y = read_dataset_to_list(train_name)
        
#        X = np.load('X_train_v3.npy')
#        y = np.load('y_train_v3.npy')
        
#        self.n_samples = X.shape[0]
#        self.x_data = torch.from_numpy(X)
#        self.y_data = torch.from_numpy(y)
#        self.x_data = self.x_data.to(torch.device("cuda"), dtype = torch.float)
#        self.y_data = self.y_data.to(torch.device("cuda"), dtype = torch.float)
        
        self.n_samples = len(X)
        
        self.x_data = X
        self.y_data = y
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
#        a = self.x_data[index][0]
#        a = np.zeros((2, 2))
#        a = "one"
#        return a, self.y_data[index]
#        print(self.x_data[index])
#        X = get_data_npy()
#        folder_name = sample_id
#        filename = sample_id + '.npy'
#        enc = get_all_parts_blip(sample_id)
        y = torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(self.y_data[index]), value=1)
        X = get_all_parts_blip(self.x_data[index])
        
        X = torch.from_numpy(X)
#        y = torch.from_numpy(y)
        X = X.to(torch.device("cuda"), dtype = torch.float)
        y = y.to(torch.device("cuda"), dtype = torch.float)
        
        return X, y

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class test_dataset(Dataset):

    def __init__(self):

        
        X,y = read_dataset_to_list(test_name)
        
        self.n_samples = len(X)
        
        self.x_data = X
        self.y_data = y
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
#        a = self.x_data[index][0]
#        a = np.zeros((2, 2))
#        a = "one"
#        return a, self.y_data[index]
#        print(self.x_data[index])
#        X = get_data_npy()
    
#        return get_all_parts_blip(self.x_data[index]), self.y_data[index]
        y = torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(self.y_data[index]), value=1)
        X = get_all_parts_blip(self.x_data[index])
        
        X = torch.from_numpy(X)
#        y = torch.from_numpy(y)
        X = X.to(torch.device("cuda"), dtype = torch.float)
        y = y.to(torch.device("cuda"), dtype = torch.float)
        
        return X, y

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


train_name = "train_split_1.csv"
test_name = "test_split_1.csv"

df_csv_file = "Dataset_uniqid_class.csv"
parent_folder = r'E:\CVPR 24\datasets\Personality dataset output\default settings'

# create dataset
train_dataset = train_dataset()


total_train_samples = len(train_dataset)
print(total_train_samples)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2,
                          shuffle=False)



test_dataset = test_dataset()


total_test_samples = len(test_dataset)
print(total_test_samples)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=2,
                          shuffle=False)


for i, (X_curr, y_curr) in enumerate(train_loader): 
    print(X_curr, y_curr)
    if (i > 2):
        break



for i, (X_curr, y_curr) in enumerate(test_loader): 
    print(X_curr, y_curr)
    if (i > 2):
        break













