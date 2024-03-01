import csv
from Final_all_parts_blip_loader import get_all_parts_blip
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly as py
#import plotly

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


traitmap = {'O':0 ,'C':1 ,'E':2 ,'A':3 ,'N':4}


train_name = "train_split_1.csv"
test_name = "test_split_1.csv"

df_csv_file = "Dataset_uniqid_class.csv"
parent_folder = r'E:\CVPR 24\datasets\Personality dataset output\default settings'



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


x,y = read_dataset_to_list(train_name)

#print(len(x))
#print(len(y))
#
#print(x[0],y[0])

#x_input = get_all_parts_blip(x[0])

#torch.empty((0, 512), dtype=torch.float32)
#print(x_input.shape)
#print(x_input.dtype)


x_array = np.empty((0, 49152), dtype=np.single)

for i in x:
    
    x_curr = x_input = get_all_parts_blip(x[0])
#    
    x_array = np.vstack((x_array,x_curr))

#    print(x_array.shape, x_array.dtype)
#    break

print(x_array.shape, x_array.dtype)

y_array = np.array(y)
print(y_array.shape, y_array.dtype)



plotX = pd.DataFrame(x_array)
#plotX.columns = y_array
plotX["Cluster"] = y_array
#print(plotX.head)


perplexity = 80

#T-SNE with one dimension
tsne_1d = TSNE(n_components=1, perplexity=perplexity)
#tsne_1d = TSNE(n_components=1)

#T-SNE with two dimensions
tsne_2d = TSNE(n_components=2, perplexity=perplexity)
#tsne_2d = TSNE(n_components=2)


#T-SNE with three dimensions
tsne_3d = TSNE(n_components=3, perplexity=perplexity)
#tsne_3d = TSNE(n_components=3)

#This DataFrame holds a single dimension,built by T-SNE
TCs_1d = pd.DataFrame(tsne_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#This DataFrame contains two dimensions, built by T-SNE
TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

#And this DataFrame contains three dimensions, built by T-SNE
TCs_3d = pd.DataFrame(tsne_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

TCs_1d.columns = ["TC1_1d"]
#PCs_1d.columns = ["PC1_1d"]
TCs_2d.columns = ["TC1_2d","TC2_2d"]
TCs_3d.columns = ["TC1_3d","TC2_3d","TC3_3d"]

print(TCs_2d.head)

plotX = pd.concat([plotX,TCs_1d,TCs_2d,TCs_3d], axis=1, join='inner')

plotX["dummy"] = 0

cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]
cluster3 = plotX[plotX["Cluster"] == 3]
cluster4 = plotX[plotX["Cluster"] == 4]

#print(cluster0)
#print(cluster1)
#print(cluster2)
#print(cluster3)
#print(cluster4)



#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["TC1_2d"],
                    y = cluster0["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["TC1_2d"],
                    y = cluster1["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["TC1_2d"],
                    y = cluster2["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 3'
trace4 = go.Scatter(
                    x = cluster3["TC1_2d"],
                    y = cluster3["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 3",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 4'
trace5 = go.Scatter(
                    x = cluster4["TC1_2d"],
                    y = cluster4["TC2_2d"],
                    mode = "markers",
                    name = "Cluster 4",
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),
                    text = None)




data = [trace1, trace2, trace3, trace4, trace5]

title = "Visualizing Clusters in Two Dimensions Using T-SNE with perplexity = " + str(perplexity)  

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)





