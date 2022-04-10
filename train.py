# -*- coding = utf-8 -*-
# @Time : 2022/4/4 15:16
# @Author : zzn
# @File : train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#####################  hyper parameters  ####################
EPISODES = 200
EP_STEPS = 200
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.001
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
RENDER = False


########################## DDPG Framework ######################
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


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0  # serves as updating the memory data
        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):  # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old data with new data
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
        # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

    def save(self):
        torch.save({
            'actor_eval_state_dict': self.actor_eval.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_eval_state_dict': self.critic_eval.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, PATH)


############################### Training ####################################
# Define the env
s_dim=5
a_dim=5
a_bound=np.array([1,1,1,1,1])
a_low_bound = np.array([-1,-1,-1,-1,-1])
PATH='D:\\大三下\\一些探索\\复兴领智\\week4\\model.pth'
DATA='data/train_test.txt'
cells=[[1,1,0,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],[0,0,0,1,1]]  # 固定细胞
cells=np.array(cells)
ddpg = DDPG(a_dim, s_dim, a_bound)
key = False
Times = 100

for time in range(Times):
    # 随机产生权重，并计算Mix
    w0 = np.random.randint(1,10,(1,5))
    Mix = np.matmul(w0, cells)
    var = 3  # the controller of exploration which will decay during training process
    for i in range(EPISODES):
        if key==False:
            w = np.random.randint(0,5,(1,5))
            observ = np.matmul(w, cells)
            s0 = observ
            s = s0.reshape(5)
            ep_r = 0
            for j in range(EP_STEPS):
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
                a = np.round(a)
                for ii in range(5):
                    if a[ii]==1:
                        w[0,ii]+=1
                    elif a[ii]==-1:
                        w[0,ii]-=1
                    else:
                        continue
                s_ = np.matmul(w, cells)
                s_ = s_.reshape(5)
                r = -sqrt(np.sum((s_ - Mix) ** 2))
                # print(r)
                ddpg.store_transition(s, a, r / 10, s_)  # store the transition to memory
                if ddpg.pointer > MEMORY_CAPACITY:
                    var *= 0.9995  # decay the exploration controller factor
                    ddpg.learn()
                s = s_
                ep_r += r
                if j == EP_STEPS - 1:
                    print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
                    with open(DATA,'a') as f:
                        f.write(str(ep_r)+"\n")
                    if ep_r > -50:
                        key=True
                    break
        else:
            break
ddpg.save()