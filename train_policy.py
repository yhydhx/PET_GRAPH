import torch
import torch.nn as nn
import numpy as np
import sys
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)
from multi_search_graph import Train
from torch.autograd import Variable
class policy(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(policy, self).__init__()
        self.linear1 = torch.nn.Linear(input_size,2000)
        self.linear2 = torch.nn.Linear(2000,800)
        self.linear3 = torch.nn.Linear(800,200)
        self.linear4 = torch.nn.Linear(200,500)
        self.linear5_add = torch.nn.Linear(500,output_size)
        self.linear5_remove = torch.nn.Linear(500,output_size)
        self.act1= nn.ReLU()
        self.act2= nn.ReLU()
        self.act3= nn.ReLU()
        self.act4= nn.ReLU()
        self.sma=nn.Softmax()
        self.smm=nn.Softmax()

    def forward(self, x):
        out= self.linear1(x)
        out = self.act1(out)
        out= self.linear2(out)
        out = self.act2(out)
        out = self.linear3(out)
        out = self.act3(out)
        out = self.linear4(out)
        out = self.act4(out)
        out_add = self.linear5_add(out)
        out_add=self.sma(out_add)
        out_remove = self.linear5_remove(out)
        out_remove=self.smm(out_remove)
        return [out_add,out_remove]


class DiaEnv():
    def __init__(self,params=1):
        self.runtrain=Train()
        self.runtrain.set_params(params)
        train_data, train_label, test_data, test_label, L, lmax, A, candidate , old_A = self.runtrain.prepare(self.runtrain.dataset)
        self.candiatelen=len(candidate)
        self.p=200.0/self.candiatelen
        obs = [np.random.choice([0,1],p=[1-self.p,self.p]) for _ in range(len(candidate))]
        self.observation = obs
    def set_params(params):
        self.runtrain.set_params(params)
    def reset(self):
        obs = [np.random.choice([0,1],p=[1-self.p,self.p]) for _ in range(self.candiatelen)]
        self.observation = obs
        return self.observation
    def step(self, action,Val=True):
        if action[0]>=0:
            self.observation[action[0]]=1
        if action[1]>=0:
            self.observation[action[1]]=0
        if Val:
            new_A = self.runtrain.action_graph(self.observation)
            reward = self.runtrain.train_one(new_A)
        else:
            reward=0
        return self.observation, reward

if __name__ == '__main__':
    Iftrain=True

    Ifloadpolicy=False
    Ifloadstate=False
    num_episode = 10000
    init_state_step=1000000
    learning_rate = 0.001
    gamma = 0.99
    batch_size=2


    env = DiaEnv(params=1)
    candiatelen=env.candiatelen
    if Ifloadpolicy:
        model=torch.load('./policy.pt')
    else:  
        model = policy(candiatelen,candiatelen).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    alpha=1.0/candiatelen

    for e in range(num_episode):
        print("episode:",e)
        ###new state
        if e%init_state_step==0:
            if Ifloadstate:
                state=torch.load('./state.pt')
            else:  
                state = env.reset()
                state = torch.from_numpy(np.array(state)).float().to(device)
                state = Variable(state).to(device)
                state.require_grad=False





        
        state_pool = []
        action_pool = []
        reward_pool = []
        for t in range(batch_size):
            [probs_add,probs_remove] = model(state.reshape([1,-1]))

            statemask=state.reshape([1,-1]).cpu().detach().numpy()[0]
            #probs_add=probs_add.cpu().detach_().numpy().reshape(-1)
            #probs_remove=probs_remove.cpu().detach_().numpy().reshape(-1)

            #####pick action
            pos_add=np.where(statemask ==0)[0]
            probs_add=probs_add[0].cpu().detach().numpy()
            p_add=probs_add[pos_add]
            p_add=p_add/p_add.sum()
            action_add=int(np.random.choice(pos_add, 1, p=p_add)[0])
            

            pos_remove=np.where(statemask ==1)[0]
            probs_remove=probs_remove[0].cpu().detach().numpy()
            p_remove=probs_remove[pos_remove]
            p_remove=p_remove/p_remove.sum()
            action_remvoe=int(np.random.choice(pos_remove, 1, p=p_remove)[0])
            ## threshold
            if alpha>probs_add[action_add]:
                action_add=-1
            if alpha>probs_remove[action_remvoe]:
                action_remvoe=-1

            
            action=[int(action_add),int(action_remvoe)]
            
            #next_state, reward, = state,1
            next_state, reward, = env.step(action,Val=Iftrain)
            print(f'Action:{action},Reward:{reward}')
            next_state=torch.from_numpy(np.array(next_state)).float().to(device)
            next_state= Variable(next_state).to(device)
            next_state.require_grad=False
        
            state_pool.append(state.clone().detach())
            action_pool.append(action)
            reward_pool.append(reward)
            state = next_state
        print('updating policy')

        # Update policy
        if Iftrain:


            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool) + 1e-9
            for i in range(batch_size):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()
            criterion = nn.NLLLoss()
            for i in range(batch_size):
                state = state_pool[i]
                reward = reward_pool[i]
                [probs_add,probs_remove] = model(state)
                target_add=Variable(torch.tensor([action_pool[i][0]])).to(device)
                target_remove=Variable(torch.tensor([action_pool[i][1]])).to(device)
                loss = criterion(probs_add.reshape(1,-1),target_add) * reward+  criterion(probs_remove.reshape(1,-1),target_remove) * reward
                loss.backward()
            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            torch.save(state, './state.pt')
            torch.save(policy,'./policy.pt')
