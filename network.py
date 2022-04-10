# -*- coding = utf-8 -*-
# @Time : 2022/4/5 20:27
# @Author : zzn
# @File : network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):  # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 400)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization of FC1
        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization of FC2
        self.out = nn.Linear(300, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initilizaiton of OUT

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x
        return actions


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs1 = nn.Linear(s_dim, 400)
        self.fcs1.weight.data.normal_(0, 0.1)

        self.fca1 = nn.Linear(a_dim, 400)
        self.fca1.weight.data.normal_(0, 0.1)

        self.fcs2 = nn.Linear(400, 300)
        self.fcs2.weight.data.normal_(0, 0.1)

        self.fca2 = nn.Linear(400, 300)
        self.fca2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(300, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs1(s)
        x = F.relu(x)
        x = self.fcs2(x)
        y = self.fca1(a)
        y = F.relu(y)
        y = self.fcs2(y)
        actions_value = self.out(F.relu(x + y))
        return actions_value