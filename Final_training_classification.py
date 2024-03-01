import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from Final_all_parts_blip_loader import get_all_parts_blip
import csv



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
#        y = torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(self.y_data[index]), value=1)
#        X = get_all_parts_blip(self.x_data[index])
#        
#        X = torch.from_numpy(X)
#        y = torch.from_numpy(y)
##        X = X.to(torch.device("cuda"), dtype = torch.float)
##        y = y.to(torch.device("cuda"), dtype = torch.float)
#        
#        return X, y
    
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
#        y = torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(self.y_data[index]), value=1)
#        X = get_all_parts_blip(self.x_data[index])
#        
#        return X, y
        
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

train_name = "train_split_70_1.csv"
test_name = "test_split_30_1.csv"

df_csv_file = "Dataset_uniqid_class.csv"
parent_folder = r'E:\CVPR 24\datasets\Personality dataset output\default settings'

# create dataset
train_dataset = train_dataset()
test_dataset = test_dataset()

#train_dataset = train_dataset()
#test_dataset = test_dataset()

# =============================================================================
# # get 4th sample and unpack
# X_curr,y_curr = dataset[4]
# print(X_curr, y_curr)
# =============================================================================

# hyper parameters
num_epochs = 25
total_samples = len(train_dataset)
batch_size = 16
n_iterations = math.ceil(total_samples/batch_size)
lr  = 0.001
lr2 = 0.0001
lr3 = 0.00001
input_size = 49152
hidden_size = 512 


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

# =============================================================================
# convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)
# =============================================================================

# =============================================================================
# examples = iter(test_loader)
# example_data, example_targets = examples.next()
# print(example_data.shape, example_targets.shape)
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(49152, 3072)
        self.fc2 = nn.Linear(3072, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
#model = Net()
model = Net().to(device)
  
# Loss and optimizer
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model.parameters(), lr=lr2)
optimizer3 = torch.optim.Adam(model.parameters(), lr=lr3)


# =============================================================================
# Train the model
# =============================================================================

n_total_steps = len(train_loader)
#print(n_total_steps)
for epoch in range(num_epochs):
    for i, (X_curr, y_curr) in enumerate(train_loader): 
        
        X_curr = X_curr.reshape(-1, input_size).to(device)
        y_curr = y_curr.to(device)

        # Forward pass
        outputs = model(X_curr)        
        loss = criterion(outputs, y_curr)
        
        # Backward and optimize
        optimizer.zero_grad()
        optimizer2.zero_grad()        
        loss.backward()
        
# =============================================================================
#         if epoch >= 10 and epoch < 20:
#             optimizer2.step()
# #            print("Used optimiser 2")
#         elif epoch >= 20:
#             optimizer3.step()
# #            print("Used optimiser 3")
#         else :
#             optimizer.step()  
# #            print("Used optimiser 1")
# =============================================================================
        optimizer.step() 
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], \tStep [{i+1}/{n_total_steps}], \tLoss: {loss.item():.10f}')

# =============================================================================
# Test Model
# =============================================================================

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        X, y = data
        output = model(X.view(-1,input_size))
        #print(output)
        for idx, i in enumerate(output):
#            print(f"{i}\n{torch.argmax(i)}\n{torch.argmax(y[idx])}")
#            print(torch.argmax(i), y[idx])
            if torch.argmax(i) == torch.argmax(y[idx]):
                correct += 1
            total += 1

acc = 100.0 * correct / total
print("Accuracy: ", round(acc, 10))

# =============================================================================
# Save model
# =============================================================================

model_name = "big5mode_" + str(int(acc)) + ".dat"
torch.save(model.state_dict(), model_name)



































