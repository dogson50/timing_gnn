from torchmetrics import R2Score
import random
from importlib.resources import path
from lib2to3.pytree import Node
from tkinter import N
from tracemalloc import start
from dataset import *
from options import get_options
from model import *
from TimeConv import *
from Unet import UNet
import dgl
import pickle
import numpy as np
import os
from time import time
from random import shuffle
import itertools
from MyDataloader import *
import tee
from torch.utils.data import DataLoader


file_path1 = sys.argv[1]
file_path2 = sys.argv[2]

def load_data(dataset_file):

    graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
    # with open(dataset_file,'rb') as f:
    # graph,topo_levels,path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = pickle.load(f)
    print(path_masks.shape, graph.ndata['cell_feat'].shape)

    #graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float)
    # if feat_reduce is not None:
    #     if feat_reduce[1] != 0:
    #         graph.ndata['net_feat'] = graph.ndata['net_feat'][:, :-feat_reduce[1]]
    #     if feat_reduce[0] != 0:
    #         graph.ndata['cell_feat'] = graph.ndata['cell_feat'][:, :-feat_reduce[0]]
        # print(graph.ndata['cell_feat'][:5])
    # normalize all the features, so that the value of different feature will not differ greatly
    # graph.ndata['cell_feat'] = transform_cellfeat(graph.ndata['cell_feat'])

    if type(cnn_inputs) == np.ndarray:
        cnn_inputs = th.from_numpy(cnn_inputs).float()
    # cnn_inputs = th.unsqueeze(cnn_inputs,dim=0)
    paths = list(range(len(graph.ndata['end'][graph.ndata['end'].squeeze() == 1])))
    # non_critical_paths = list(set(paths)-set(critical_paths))
    num_neg = len(paths) - len(critical_paths)
    num_pos = len(critical_paths)
    ratio = num_neg / num_pos - 1
    return (paths, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks)


paths1, graph1, path2level1, path2endpoint1, topo_levels1, cnn_inputs1, path_masks1 = load_data(file_path1)
paths2, graph2, path2level2, path2endpoint2, topo_levels2, cnn_inputs2, path_masks2 = load_data(file_path2)
nfeat1 = graph1.ndata['net_feat']
cfeat1 = graph1.ndata['cell_feat']
nfeat2 = graph2.ndata['net_feat']
cfeat2 = graph2.ndata['cell_feat']
rt1 = graph1.ndata['required_time']
at1 = graph1.ndata['arrival_time']
rt2 = graph2.ndata['required_time']
at2 = graph2.ndata['arrival_time']
lb1 = graph1.ndata['label']
lb2 = graph2.ndata['label']
print("path:",paths1==paths2)
print("graph:",graph1==graph2)
print("graph label:",len(lb1[lb1!=lb2])==0)
print("graph nfeat:",len(nfeat1[nfeat1!=nfeat2])==0)
print("graph cfeat:",len(cfeat1[cfeat1!=cfeat2])==0)
print("graph required time:",len(rt1[rt1!=rt2])==0)
print("graph arrival time:",len(at1[at1!=at2])==0)
print("path2level:",path2level1==path2level2)
print("path2endpoint:",path2endpoint1==path2endpoint2)
print("cnn_inputs:",len(cnn_inputs1[cnn_inputs1!=cnn_inputs2])==0)
print("path_masks:",len(path_masks1.to_dense()[path_masks1.to_dense()!=path_masks2.to_dense()])==0)
flag = True
for i in range(len(topo_levels1)):
    level1 = topo_levels1[i]
    level2 = topo_levels2[i]
    level1[0].sort()
    level2[0].sort()
    level1[1].sort()
    level2[1].sort()
    level1[2].sort()
    level2[2].sort()
    if level1!=level2:
        print('level',i)
        print('\t',level1)
        print('\t', level2)
        flag = False
        break
print("topo_levels:",flag)

#print("path_masks:",path_masks1.to_dense()==path_masks2.to_dense())
