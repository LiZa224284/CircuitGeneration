import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.datasets import get_dataset,get_circuit_dataset,MyOwnDataset
# from kernel.train_eval import cross_validation_with_val_set
# from kernel.train_eval import cross_validation_without_val_set
from kernel.train_eval import trainFEGIN
# from kernel.gcn import *
# from kernel.graph_sage import *
# from kernel.gin import *
from kernel.FEGIN import *
# from kernel.gat import *
# from kernel.graclus import Graclus
# from kernel.top_k import TopK
# from kernel.diff_pool import *
# from kernel.global_attention import GlobalAttentionNet
# from kernel.set2set import Set2SetNet
# from kernel.sort_pool import SortPool
import warnings
import argparse
import netlsd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.utils import shuffle
from torch_geometric.data import DenseDataLoader as DenseLoader
from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
from torch.optim import Adam

device = 'mps'


# import and preprocess the dataset _____________________ generate: processsed/ltspice_examples_dataset.pt    pre_filter.pt   pre_transform.pt
dataset_name = "ltspice_examples"

dataset = MyOwnDataset("MergeTest/data/", dataset_name, 
                h = 12, # the height of rooted subgraph for NGNN models
                node_label = "spd", 
                use_rd = None, 
                max_nodes_per_hop = None)


# porcess the dataset to be compatible with the models _____________________ generate: MergeTest/data/ltspice_examples_netlsd_train.pt    MergeTest/data/ltspice_examples_netlsd_test.pt

batch_size=128
if os.path.isfile('MergeTest/data/'+dataset_name+'_netlsd_train.pt'):
        train_des = torch.load('MergeTest/data/'+dataset_name+'_netlsd_train.pt')
        test_des = torch.load('MergeTest/data/'+dataset_name+'_netlsd_test.pt')
        train_dataset = [d for d in dataset if d.set =='train']
        test_dataset = [d for d in dataset if d.set =='test']
else:
    train_dataset,train_des, test_dataset, test_des = [],[],[],[]
    for index, d in enumerate(dataset):
        if d.set=='train':
            des = (torch.tensor(netlsd.heat(to_networkx(d, to_undirected = True)))*0.1).float()
            des_d = Data(x = des, edge_index = d.edge_index, y = d.y)
            train_dataset.append(d)
            train_des.append(des_d)
            # if index%50==0:
            #     print(index)

        else:
            des = (torch.tensor(netlsd.heat(to_networkx(d, to_undirected = True)))*0.1).float()
            des_d = Data(x = des, edge_index = d.edge_index, y = d.y)
            test_des.append(des_d)
            test_dataset.append(d)

    torch.save(train_des,'MergeTest/data/'+dataset_name+'_netlsd_train.pt')
    torch.save(test_des,'MergeTest/data/'+dataset_name+'_netlsd_test.pt')

 #   porcess the dataset to be compatible with the models_____________________ generate: 
train_dataset, train_des = shuffle(train_dataset, train_des)


if 'adj' in train_dataset[0]:
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
else:
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_loader_des = DataLoader(train_des, batch_size, shuffle=False)
    test_loader_des = DataLoader(test_des, batch_size, shuffle=False)


 #   porcess the dataset to be compatible with the models_____________________ generate: 
Net = FEGIN
model = Net(dataset, num_layers=4, hidden=32, emb_size=250,node_label = "spd", use_rd=False)
model.to(device).reset_parameters()
model.train()
optimizer = Adam(model.parameters(), lr=1E-2, weight_decay=0)
loader = train_loader
des_loader = train_loader_des

for data,des in zip(loader,des_loader):
        optimizer.zero_grad()
        data = data.to(device)
        des = des.to(device)
        out = model(data,des)

  
# loss, f1,f1_std = trainFEGIN(
#                     dataset,dataset_name,
#                     model,
#                     folds=3,
#                     epochs=1,
#                     batch_size=128,
#                     lr=1E-2,
#                     lr_decay_factor=0.5,
#                     lr_decay_step_size=50,
#                     weight_decay=0,
#                     device=device, 
#                     logger=None)


