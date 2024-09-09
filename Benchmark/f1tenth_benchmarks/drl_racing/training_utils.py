import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NN_LAYER_1 = 100
# NN_LAYER_2 = 100
# NN_LAYER_1 = 400
# NN_LAYER_2 = 300

NN_LAYER_1 = 256
NN_LAYER_2 = 256

MEMORY_SIZE = 100000

class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr
    

class TinyPolicyBuffer(object):
    def __init__(self, state_dim, action_dim, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.memory_size = memory_size

        self.states = np.empty((memory_size, *state_dim))
        self.actions = np.empty((memory_size, action_dim))
        self.next_states = np.empty((memory_size, *state_dim))
        self.rewards = np.empty((memory_size, 1))
        self.dones = np.empty((memory_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == self.memory_size:
            self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, *self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, *self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1 - dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr
   


class TinyPolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(TinyPolicyNet, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 16, 4, 2)
        self.conv2 = nn.Conv1d(16, 32, 2, 1)
        self.fc1 = nn.Linear(256, 64)
        self.fc_mu = nn.Linear(64, act_dim)

    def forward(self, x):
        # print(x.shape)
        if (x.dim() == 2):
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]


class TinyCriticNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(TinyCriticNet, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 16, 4, 2)
        self.conv2 = nn.Conv1d(16, 32, 2, 1)
        self.fc1 = nn.Linear(256 + act_dim, 64)
        self.fc_mu = nn.Linear(64, act_dim)

    def forward(self, state, action):
        # print(state.shape, action.shape)
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = torch.flatten(state, 1)
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        q = self.fc_mu(x2)
        return q
        

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetworkSAC, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
        log_prob = log_prob.sum(-1, keepdim=True)
            
        return action, log_prob
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.forward(state)
        return action.detach().numpy()[0]
   
class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]


class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        #print(state.shape, action.shape)
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q




if __name__ == "__main__":
    print("Hello World!")
    
    