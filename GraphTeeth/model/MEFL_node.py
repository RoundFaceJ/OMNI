import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .graph import create_e_matrix, normalize_digraph
from .graph_edge_model import GEM
from .basic_block import *

class GNN_Node(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=5, metric='dots'):
        super(GNN_Node, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':  
            si = x.detach()  
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))  
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)  
            adj = (si >= threshold).float()  
        
        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)
        
        # GNN process
        A = normalize_digraph(adj)  

        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))  
        x = self.relu(x + self.bnv(aggregate + self.U(x)))  
        return x 

class Head(nn.Module):
    def __init__(self, in_channels, num_node, num_classes, neighbor_num):
        super(Head, self).__init__()

        self.in_channels = in_channels
        self.num_node = num_node
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        
        self.gnn_node = GNN_Node(self.in_channels, self.num_node, neighbor_num) 


    def forward(self, global_features, box_features):
        
        # only node
        batch_size = global_features.shape[0]
        box_features = box_features.reshape(batch_size, box_features.shape[0] // batch_size, box_features.shape[1], box_features.shape[2])
        f_v = box_features.mean(dim=-2)
        f_v = self.gnn_node(f_v)
        cl = self.fc(f_v.view(-1, f_v.shape[2]))
        
        return cl


class MEFARG(nn.Module):
    def __init__(self, num_node=50, num_classes=11, neighbor_num=5):
        super(MEFARG, self).__init__()
        self.linear3 = nn.Linear(256, 512)

        self.head = Head(512, num_node, num_classes, neighbor_num)


    def forward(self, x):

        
        
        
        global_features, box_features = x
        global_features = global_features['3'] # P5 level features

        
        b, c, h, w = box_features.shape
        box_features = box_features.view(b, c, -1).permute(0,2,1) 
        box_features = self.linear3(box_features)
        
        cl = self.head(global_features, box_features)
        return cl

