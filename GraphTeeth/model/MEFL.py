import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .graph import create_e_matrix, normalize_digraph
from .graph_edge_model import GEM
from .basic_block import *


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):  
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out

        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)  
        e = self.softmax(e)  
        e = e.view(b, -1, c)  


        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out 
        x = self.act(res + self.bnv1(x))  
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge  # [32, 12, 512]  [32, 144, 512]

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
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n,
                                                                                             1)  
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
        x = self.relu(
            x + self.bnv(aggregate + self.U(x)))  
        return x  

class Head(nn.Module):
    def __init__(self, in_channels, num_node, num_classes, neighbor_num):
        super(Head, self).__init__()

        self.in_channels = in_channels
        self.num_node = num_node
        self.num_classes = num_classes
        self.edge_extractor = GEM(self.in_channels, self.num_node)
        self.gnn = GNN(self.in_channels, self.num_node)
        self.fc = nn.Linear(self.in_channels, self.num_classes)


    def forward(self, global_features, box_features):
        
        # use edge
        batch_size = global_features.shape[0]
        box_features = box_features.reshape(batch_size, box_features.shape[0] // batch_size, box_features.shape[1], box_features.shape[2])
        f_e = self.edge_extractor(box_features, global_features)
        f_v = box_features.mean(dim=-2)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.gnn(f_v, f_e)    
        
        cl = self.fc(f_v.view(-1, f_v.shape[2]))

        return cl


class MEFARG(nn.Module):
    def __init__(self, num_node=50, num_classes=11, neighbor_num=5):
        super(MEFARG, self).__init__()
        
        self.linear1 = nn.Linear(1050, 49)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(256, 512)

        self.head = Head(512, num_node, num_classes, neighbor_num)


    def forward(self, x):

        
        
        
        global_features, box_features = x
        global_features = global_features['3'] # P5 level features
        b, c, h, w = global_features.shape
        global_features = global_features.view(b, c, -1)
        global_features = self.linear1(global_features).permute(0,2,1) 
        global_features = self.linear2(global_features)  
        
        b, c, h, w = box_features.shape
        box_features = box_features.view(b, c, -1).permute(0,2,1) 
        box_features = self.linear3(box_features)
        
        cl = self.head(global_features, box_features)
        return cl

