import csv
#from pathlib import Path
#import numpy as np
import random


df_csv_file = "Dataset_uniqid_class.csv"


with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

#print(data[0:5])
#print(len(data))

random.shuffle(data)

#print(data[0:5])
#print(len(data))

split = (int) (0.9 * len(data))
train_list = data[0:split]
test_list = data[split:]

#print(type(split))
print(len(train_list))
print(len(test_list))

#for i in train_list:
#    print(i,i[0],i[1])
#    break


train_name = "train_split_1.csv"
test_name = "test_split_1.csv"

#Store train list and test list

f = open(train_name,"w")

for i in train_list:
    filewrite = '"' + i[0] + '"' + "," + '"' + i[1] + '"' + "\n" 
    f.write(filewrite)   

f.close()

f = open(test_name,"w")

for i in test_list:
    filewrite = '"' + i[0] + '"' + "," + '"' + i[1] + '"' + "\n" 
    f.write(filewrite)   

f.close()

















#with open(train_name, 'w') as csvfile:
#    csvwriter = csv.writer(csvfile) 
#    csvwriter.writerows(train_list) 
#
#with open(test_name, 'w') as csvfile:
#    csvwriter = csv.writer(csvfile) 
#    csvwriter.writerows(test_list) 


#with open(train_name, 'w' , newline='\n', encoding='utf-8') as csv_file:
#    cf = csv.writer(csv_file, delimiter=',',quotechar='"')
#    cf.writerows(train_list)
