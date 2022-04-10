# -*- coding = utf-8 -*-
# @Time : 2022/4/4 23:24
# @Author : zzn
# @File : predict.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import ActorNet
from network import CriticNet
from DDPG import DDPG
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
var = 3  # the controller of exploration which will decay during training process

############################### Load ######################################
s_dim=5
a_dim=5
a_bound=np.array([1,1,1,1,1])
a_low_bound = np.array([-1,-1,-1,-1,-1])
PATH='D:\\大三下\\一些探索\\复兴领智\\week4\\model\\model.pth'
# w_file="data/data2_w_t2.txt"
# r_file="data/data2_t2.txt"
w_file="data/test_w.txt"
w0_file='data/w0.txt'
r_file="data/test.txt"
cells=[[1,1,0,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],[0,0,0,1,1]]  # 固定细胞
cells=np.array(cells)


actor_eval = ActorNet(s_dim, a_dim)
actor_target = ActorNet(s_dim, a_dim)
critic_eval = CriticNet(s_dim, a_dim)
critic_target = CriticNet(s_dim, a_dim)
actor_optimizer = torch.optim.Adam(actor_eval.parameters(), lr=LR_ACTOR)
critic_optimizer = torch.optim.Adam(critic_eval.parameters(), lr=LR_CRITIC)

checkpoint = torch.load(PATH)
actor_eval.load_state_dict(checkpoint['actor_eval_state_dict'])
actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
critic_eval.load_state_dict(checkpoint['critic_eval_state_dict'])
critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

actor_eval.eval() # or modelA.train()
actor_target.eval() # or modelB.train()
critic_eval.eval() # or modelA.train()
critic_target.eval() # or modelB.train()

ddpg=DDPG(a_dim,s_dim,a_bound)
ddpg.actor_eval=actor_eval
ddpg.actor_target=actor_target
ddpg.critic_eval=critic_eval
ddpg.critic_target=critic_target
ddpg.actor_optimizer=actor_optimizer
ddpg.critic_optimizer=critic_optimizer



# ################################ Predict ######################################
Times=100
w0=[]
with open(w0_file, 'r') as f:
    lines=f.readlines()
    for line in lines:
        data=line.replace('\n','').replace('[','').replace(']','')
        data=data.split(' ')
        data=np.array(data,dtype=np.int).reshape((1,5))
        w0.append(data)

for i in range(Times):
    Mix = np.matmul(w0[i], cells)
    w = np.random.randint(1, 5, (1, 5))
    observ = np.matmul(w, cells)
    s0 = observ
    s = s0.reshape(5)
    ep_r = 0
    key = False
    j = 0

    while key == False:
        j += 1
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
        ddpg.store_transition(s, a, r / 10, s_)  # store the transition to memory
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= 0.9995  # decay the exploration controller factor
            ddpg.learn()
        s = s_
        ep_r += r
        if j%100 == 0:
            # ep_r=ep_r*1000/j
            print('Episode: ', j//100, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)

            with open(r_file, 'a') as f:
                f.write(str(r) + "\n")
            if ep_r > -95:
                key = True
                with open(w_file, 'a') as f:
                    f.write("w is : " + str(w) + "\n")
            ep_r = 0