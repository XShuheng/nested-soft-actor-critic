'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''


import math
import random
import time
#import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
#from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
#from IPython.display import display

from Power_Env import IEGS, IEGS1
#from Operation_Env import IEGS
import argparse
import time

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if len(self.buffer) > 2000:
            del(self.buffer[0])
            self.buffer[1999] = (state, action, reward, next_state, done)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)




class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)/24
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()




class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return float(predicted_new_q_value.mean()), float(q_value_loss1), float(q_value_loss2), float(
            policy_loss),float(alpha_loss),float(self.alpha)
        #return float(predicted_new_q_value.mean()),float(q_value_loss1),float(q_value_loss2),float(policy_loss),float(predicted_q_value1),float(predicted_q_value2)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


class ReplayBuffer_c:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push_c(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if len(self.buffer) > 2000:
            del(self.buffer[0])
            self.buffer[1999] = (state, action, reward, next_state, done)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        #self.buffer[self.position] = (state, action, reward, next_state, done)
        #self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork_c(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork_c, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, num_actions)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        x = self.linear4(x)
        return x



class PolicyNetwork_c(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork_c, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions

    def forward(self, state, softmax_dim=-1):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = F.tanh(self.linear4(x))

        probs = F.softmax(self.output(x), dim=softmax_dim)

        return probs

    def evaluate(self, state, epsilon=1e-8):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.forward(state, softmax_dim=-1)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action

class SAC_Trainer_c():
    def __init__(self, replay_buffer, hidden_dim):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork_c(state_dim_c, action_dim_c, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork_c(state_dim_c, action_dim_c, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork_c(state_dim_c, action_dim_c, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork_c(state_dim_c, action_dim_c, hidden_dim).to(device)
        self.policy_net = PolicyNetwork_c(state_dim_c, action_dim_c, hidden_dim).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def get_q_value(self, state,action):
        state = torch.FloatTensor(state).to(device)
        #action = torch.FloatTensor(action).to(device)
        action = torch.Tensor(action).to(torch.int64).to(device)
        predicted_q_value1 = self.soft_q_net1(state)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        return float(predicted_q_value1)

    def update_c(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.Tensor(action).to(torch.int64).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        predicted_q_value1 = self.soft_q_net1(state)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        predicted_q_value2 = self.soft_q_net2(state)
        predicted_q_value2 = predicted_q_value2.gather(1, action.unsqueeze(-1))
        log_prob = self.policy_net.evaluate(state)
        with torch.no_grad():
            next_log_prob = self.policy_net.evaluate(next_state)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        self.alpha = self.log_alpha.exp()
        target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(next_state), self.target_soft_q_net2(
            next_state)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        with torch.no_grad():
            predicted_new_q_value = torch.min(self.soft_q_net1(state), self.soft_q_net2(state))
        policy_loss = (log_prob.exp() * (self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # print('q loss: ', q_value_loss1.item(), q_value_loss2.item())
        # print('policy loss: ', policy_loss.item() )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return float(predicted_new_q_value.mean()), float(q_value_loss1), float(q_value_loss2), float(
            policy_loss),float(alpha_loss),float(self.alpha)

        #return float(predicted_new_q_value.mean()),float(q_value_loss1),float(q_value_loss2),float(policy_loss),float(predicted_q_value1),float(predicted_q_value2)


    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()



def plot(rewards):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    # plt.show()


replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
replay_buffer_c = ReplayBuffer_c(replay_buffer_size)

# choose env
ENV = ['Reacher', 'Pendulum-v0', 'HalfCheetah-v2'][0]

action_range = 1
env=IEGS1()
action_dim = env.action_space
state_dim  = env.observation_space
action_dim_c = 20
state_dim_c  = 66

# hyper-parameters for RL training
max_episodes  = 1000
max_steps = 24

frame_idx   = 0
batch_size  = 1200
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
replay_num=84000000
rewards     = []
rewards2    = []
loss11     = []
loss12    = []
loss13     = []
loss14     = []
loss15    = []
loss16     = []
loss21    = []
loss22     = []
loss23    = []
loss24    = []
loss25     = []
loss26    = []
rewards2t     = []
rewards1t     = []
rewards0t     = []



rewards_o     = []
rewards2_o    = []
rewards2t_o     = []
rewards1t_o     = []
rewards0t_o     = []
model_path = './model/sac_v2'
model_path1 = './model/sac_v2_1'

sac_trainer=SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range  )
sac_trainer_c=SAC_Trainer_c(replay_buffer_c, hidden_dim=hidden_dim  )



aastate = np.zeros(shape=(66))#observation_space 42+action_space 21
state = np.zeros(shape=(43))#
next_state = np.zeros(shape=(43))#
next_aastate = np.zeros(shape=(66))
next_action=np.zeros(shape=(1))
next_action[0]=-100
eli_state=np.zeros(shape=(24,43))
eli_action=np.zeros(shape=(24,24))
eli_reward=np.ones(shape=(24))*(-1e20)
eli_reward1t=np.ones(shape=(24))*(-1e20)
eli_reward0t=np.zeros(shape=(24))
eli_nstate=np.zeros(shape=(24,43))

eli_astate=np.zeros(shape=(24,66))
eli_attack=np.zeros(shape=(24,1))
eli_nastate=np.zeros(shape=(24,66))
eli_reward2=np.ones(shape=(24))*1e10
eli_reward2t=np.ones(shape=(24))*1e10
qvalue=np.zeros(shape=(1,66))
qaction=np.zeros(shape=(1))
eli_record = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

mark=0



if __name__ == '__main__':
    T1=time.time()
    title1 = 'rewards_result_episode'
    if args.train:
        record1 = []
        reward1 = []
        Inistate, action_data, data, action_basic, load_shedding,minPl = env.reset()
        aa=0
        bb=0
        aaa=0
        ccc=0
        for eps in range(max_episodes):
            episode_reward = 0
            state[0:42] = Inistate[:, 0]
            if mark==0:
                state[42] = 0
                mark+=1
            else:
                state[42] = predicted_q_value1
                #state[42] = 0
            #state[42] = 0

            episode_reward2 = 0
            episode_reward2_t = 0
            episode_reward1_t = 0
            episode_reward0_t = 0
            
            
            for step in range(max_steps):




                if next_action[0] != -100:
                    action = next_action
                else:
                    action = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)

                #print(action)
                aastate[0:42] = state[0:42]
                aastate[42:66] = action
                attack = sac_trainer_c.policy_net.get_action(aastate, deterministic=DETERMINISTIC)
                qvalue[0,:]=aastate
                qaction[0]=attack
                predicted_q_value1= sac_trainer_c.get_q_value(qvalue,qaction)
                #predicted_q_value1 = 0
                #predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
                #predicted_q_value1=1

                next_state[0:42], reward, record, done, reward2, reward2_t, reward11t, reward00t = env.step(action,action_data, data,action_basic,Inistate[:,step + 1],state, step,load_shedding,attack, minPl, 1)
                    # env.render()
                next_state[42]=predicted_q_value1
                #next_state[42] = 0
                reward1.append(reward)





                next_action = sac_trainer.policy_net.get_action(next_state, deterministic=DETERMINISTIC)
                next_aastate[0:42] = next_state[0:42]
                next_aastate[42:66] = next_action
                if len(replay_buffer) > replay_num:
                    # if eli_reward1t[step] <1000**(500/eps+0) + reward11t:
                    if eli_reward1t[step] < 1080 + reward11t:
                        if eli_reward1t[step] < reward11t:
                            for i in range(len(state)):
                                eli_state[step][i] = state[i]
                                eli_nstate[step][i] = next_state[i]
                            for i in range(len(action)):
                                eli_action[step][i] = action[i]
                            eli_reward[step] = reward
                            eli_reward1t[step] = reward11t
                            eli_reward0t[step] = reward00t

                            for i in range(len(aastate)):
                                eli_astate[step][i] = aastate[i]
                                eli_nastate[step][i] = next_aastate[i]
                            eli_attack[step][0] = attack
                            eli_reward2[step] = float(reward2)
                            eli_reward2t[step] = reward2_t
                            eli_record[step]=record

                        replay_buffer.push(state, action, reward, next_state, done)
                        replay_buffer_c.push_c(aastate, attack, float(reward2), next_aastate, done)
                        record1.append(record)

                    else:
                        reward = eli_reward[step]
                        reward11t = eli_reward1t[step]
                        reward00t = eli_reward0t[step]
                        record = eli_record[step]
                        replay_buffer.push(eli_state[step], eli_action[step], eli_reward[step], eli_nstate[step], done)

                        reward2 = eli_reward2[step]
                        reward2_t = eli_reward2t[step]
                        attack=np.array(int(eli_attack[step]))
                        replay_buffer_c.push_c(eli_astate[step], attack, eli_reward2[step], eli_nastate[step],
                                             done)
                        record1.append(record)

                else:
                    record1.append(record)
                    replay_buffer.push(state, action, reward, next_state, done)
                    replay_buffer_c.push_c(aastate, attack, reward2, next_aastate, done)

                state = next_state
                episode_reward += reward
                episode_reward2 += reward2
                episode_reward2_t += reward2_t
                episode_reward1_t += reward11t
                episode_reward0_t += reward00t
                frame_idx += 1
                

                #if len(replay_buffer) > 6000:

                #    _=sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
                #if len(replay_buffer) > batch_size:
                if aa > batch_size:
                    #if aa<2400 and bb==24:


                #        _1,__1,___1,____1,_____1,______1=sac_trainer.update(128, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
                        #__ = sac_trainer_c.update_c(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim_c)
                #        loss11.append(_1)
                #        loss12.append(__1)
                #        loss13.append(___1)
                #        loss14.append(____1)
                #        loss15.append(_____1)
                #        loss16.append(______1)

                    #if aa>128:
                #        _2,__2,___2,____2,_____2,______2 = sac_trainer_c.update_c(128, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                #                                    target_entropy=-1. * action_dim_c)
                #        loss21.append(_2)
                #        loss22.append(__2)
                #        loss23.append(___2)
                #        loss24.append(____2)
                #        loss25.append(_____2)
                #        loss26.append(______2)
                #        bb = 0
                    if aa>batch_size:
                        aaa += 1
                        if aaa > 12 or ccc > 200:
                            ccc += 1
                            aaa = 0
                    #elif aa > batch_size:
                            _1, __1, ___1, ____1, _____1, ______1 = sac_trainer.update(128, reward_scale=10.,
                                                                                       auto_entropy=AUTO_ENTROPY,
                                                                                       target_entropy=-1. * action_dim)
                            # __ = sac_trainer_c.update_c(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim_c)
                            loss11.append(_1)
                            loss12.append(__1)
                            loss13.append(___1)
                            loss14.append(____1)
                            loss15.append(_____1)
                            loss16.append(______1)

                            # if aa>128:
                            _2, __2, ___2, ____2, _____2, ______2 = sac_trainer_c.update_c(128, reward_scale=10.,
                                                                                           auto_entropy=AUTO_ENTROPY,
                                                                                           target_entropy=-1. * action_dim_c)
                            loss21.append(_2)
                            loss22.append(__2)
                            loss23.append(___2)
                            loss24.append(____2)
                            loss25.append(_____2)
                            loss26.append(______2)
                        bb+=1
                    #aa=0



                aa+=1


                if done:
                    break

            #if eps % 20 == 0 and eps>0: # plot and model saving interval
                #plot(rewards)
            #    np.save('rewards', rewards)
            #    sac_trainer.save_model(model_path)
            rewards.append(episode_reward)
            rewards2.append(episode_reward2)
            rewards2t.append(episode_reward2_t)
            rewards1t.append(episode_reward1_t)
            rewards0t.append(episode_reward0_t)
        sac_trainer.save_model(model_path)
        sac_trainer_c.save_model(model_path1)
    T2 = time.time()
    T3=T2-T1

    np.save('record_13' + '.npy', np.array(rewards))

    title = 'record_13'  # 读取的文件名
    data = np.load(title + '.npy', allow_pickle=True)  # 读取numpy文件
    data_df = pd.DataFrame(data)  # 利用pandas库对数据进行格式转换

    # print(np.linspace(1, 20, 20),draw[0])
    # plt.plot(np.linspace(1, 20, 20), draw[0])

    # create and writer pd.DataFrame to excel
    writer = pd.ExcelWriter(title1 + '.xlsx')  # 生成一个excel文件
    data_df.to_excel(writer, 'page_1')  # 数据写入excel文件
    writer.close()  # 保存excel文件




