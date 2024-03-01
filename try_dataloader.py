#import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import math
#import torch.nn as nn
#import torch.nn.functional as F
import csv

traitmap = {'O':0 ,'C':1 ,'E':2 ,'A':3 ,'N':4}

def read_dataset_to_list():
    X_list = []
    y_list = []
    with open("Dataset_uniqid_class.csv", newline='\n', encoding='utf-8') as csv_file:
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
        
        X,y = read_dataset_to_list()
        
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
        print(self.x_data[index])
#        X = get_data_npy()
    
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
train_dataset = train_dataset()


total_samples = len(train_dataset)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2,
                          shuffle=False)

for i, (X_curr, y_curr) in enumerate(train_loader): 
    print(i, X_curr, y_curr)
    if (i > 2):
        break

















