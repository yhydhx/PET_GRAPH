import torch
import torch.nn as nn
import numpy as np
import sys

import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


import imp
import importlib
device = torch.device('cuda')

import copy

from multi_search_graph import Train
from Siren import Siren

class DiaEnv():
    def __init__(self,params=1,p=200.0/1555):
        self.range = 1000  # Randomly selected number is within +/- this value
        self.bounds = 10000
        self.runtrain=Train()
        self.p=p
        self.runtrain.set_params(params)
        train_data, train_label, test_data, test_label, L, lmax, A, candidate , old_A = self.runtrain.prepare(self.runtrain.dataset)
        self.candiatelen=len(candidate)
        obs = [np.random.choice([0,1],p=[1-p,p]) for _ in range(len(candidate))]
        self.observation = obs
    def set_params(params):
        self.runtrain.set_params(params)

    def reset(self):
        obs = [np.random.choice([0,1],p=[1-self.p,self.p]) for _ in range(self.candiatelen)]
        self.observation = obs
        return self.observation 
        
    def step(self, action,Val=True):
        self.observation[action[0]]=1
        self.observation[action[1]]=0
        if Val:
            new_A = self.runtrain.action_graph(action)
            reward = self.runtrain.train_top3(new_A)
        else:
            reward=0
        return self.observation, reward

def randompick(probs):
    pd=np.random.rand()
    for i in range(len(probs)):
        pd=pd-probs[i]
        if pd<0:
            break
    pick=i-1
    return pick


from torch.autograd import Variable
from itertools import count
episode_durations = []

#plt.pause(0.001)  # pause a bit so that plots are updated

# Parameters
Iftrain=True
Ifloadpolicy=False
Ifloadstate=False
num_episode = 10
batch_size = 10
learning_rate = 0.01
gamma = 0.99

env = DiaEnv(params=1)
le=env.candiatelen
print(111,le)
if Ifloadpolicy:
    policy_net_add=torch.load('./rl/policy_net_add8.pt')
    policy_net_remove=torch.load('./rl/policy_net_remove8.pt')
else:  
    policy_net_add = Siren(in_features=le, out_features=le, hidden_features=500, 
                    hidden_layers=3, outermost_linear=True).to(device)
    policy_net_remove = Siren(in_features=le, out_features=le, hidden_features=500, 
                    hidden_layers=3, outermost_linear=True).to(device)


if Ifloadstate:
    state=torch.load('./state9.pt')
else:  
    state = env.reset()
    state = torch.from_numpy(np.array(state)).float().to(device)
    state = Variable(state).to(device)


optimizer_add = torch.optim.RMSprop(policy_net_add.parameters(), lr=learning_rate)
optimizer_remove = torch.optim.RMSprop(policy_net_add.parameters(), lr=learning_rate)


for e in range(num_episode):
    print("episode",e)
    state_pool = []
    action_pool = []
    reward_pool = []
    for t in range(batch_size):
        probs_add = policy_net_add(state.reshape([1,-1])).to(device)
        probs_add_np=probs_add.cpu().detach_().numpy().reshape(-1)
        action1  = randompick(probs_add_np)


        probs_remove = policy_net_remove(state.reshape([1,-1])).to(device)
        probs_remove_np=probs_remove.cpu().detach_().numpy().reshape(-1)
        action0  = randompick(probs_remove_np)
        action=[int(action1),int(action0)]
        print(action)
        #next_state, reward, = state,1
        next_state, reward, = env.step(action,Val=Iftrain)
        next_state=torch.from_numpy(np.array(next_state)).float().to(device)
        next_state= Variable(next_state).to(device)
        state_pool.append(state)
        action_pool.append(action)
        reward_pool.append(reward)
        state = next_state
    print('Reward',reward_pool)

    # Update policy
    if e > 0 and e % batch_size == 0 and Iftrain:


        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(batch_size):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        optimizer_add.zero_grad()
        optimizer_remove.zero_grad()
        criterion = nn.NLLLoss()
        for i in range(batch_size):
            state = state_pool[i]
            reward = reward_pool[i]
            probs_add = policy_net_add(state)
            probs_remove = policy_net_remove(state)
            target_add=Variable(torch.tensor([action_pool[i][0]])).to(device)+1
            target_remove=Variable(torch.tensor([action_pool[i][1]])).to(device)+1
            loss_add = criterion(probs_add.reshape(1,-1),target_add) * reward  # Negtive score function x reward
            loss_remove = criterion(probs_remove.reshape(1,-1),target_remove) * reward
            loss_add.backward()
            loss_remove.backward()

        optimizer_add.step()
        optimizer_remove.step()

        state_pool = []
        action_pool = []
        reward_pool = []
        torch.save(state, './rl/state10.pt')
        torch.save(policy_net_add,'./rl/policy_net_add10.pt')
        torch.save(policy_net_remove,'./rl/policy_net_remove10.pt')
