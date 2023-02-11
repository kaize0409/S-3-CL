import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np



class GNNLayer(Module):
    def __init__(self, in_features, out_features, PReLU=False):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.PReLu = PReLU
        if PReLU:
            self.act = nn.PReLU()

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            if self.PReLu:
                output = self.act(output)
            else:
                output = F.relu(output)
        return output

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):

        h_1 = F.relu(self.fc_1(x))

        h_2 = self.fc_2(h_1)

        return h_2

class simple_encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(simple_encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        output = torch.mm(features, self.weight)
        output = F.relu(output)
        return output

class S3CL_Model(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(S3CL_Model, self).__init__()
        self.encoder = simple_encoder(in_dim, out_dim)
        self.encoder_momt = simple_encoder(in_dim, out_dim)
        self.projector = MLP(in_dim, hidden_dim, out_dim)
        self.projector_momt = MLP(in_dim, hidden_dim, out_dim)
    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update 
        """
        for param_ori, param_momt in zip(self.encoder.parameters(), self.encoder_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
        for param_ori, param_momt in zip(self.projector.parameters(), self.projector_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
    def forward(self, x):
        h = self.encoder(x)
        h_p = self.projector(h)
        with torch.no_grad():  
            self._momentum_update()  
            h_momt = self.encoder_momt(x)
            h_p_momt = self.projector_momt(h_momt)
        return h, h_p, h_p_momt