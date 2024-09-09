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
    def __init__(self, actor_dim, critic_dim, action_dim):     
        self.ptr = 0
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.action_dim = action_dim

        self.actor_states = np.empty((MEMORY_SIZE, actor_dim))
        self.critic_states = np.empty((MEMORY_SIZE, critic_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_actor_states = np.empty((MEMORY_SIZE, actor_dim))
        self.next_critic_states = np.empty((MEMORY_SIZE, critic_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, actor_state, critic_state, action, next_actor_state, next_critic_state, reward, done):
        self.actor_states[self.ptr] = actor_state
        self.critic_states[self.ptr] = critic_state
        self.actions[self.ptr] = action
        self.next_actor_states[self.ptr] = next_actor_state
        self.next_critic_states[self.ptr] = next_critic_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        actor_states = np.empty((batch_size, self.actor_dim))
        critic_states = np.empty((batch_size, self.critic_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_actor_states = np.empty((batch_size, self.actor_dim))
        next_critic_states = np.empty((batch_size, self.critic_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            actor_states[i] = self.actor_states[j]
            critic_states[i] = self.critic_states[j]
            actions[i] = self.actions[j]
            next_actor_states[i] = self.next_actor_states[j]
            next_critic_states[i] = self.next_critic_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        actor_states = torch.FloatTensor(actor_states).unsqueeze(1)
        critic_states = torch.FloatTensor(critic_states).unsqueeze(1)
        actions = torch.FloatTensor(actions)
        next_actor_states = torch.FloatTensor(next_actor_states)
        next_critic_states = torch.FloatTensor(next_critic_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)

        return actor_states, critic_states, actions, next_actor_states, next_critic_states, rewards, dones

    def size(self):
        return self.ptr
   

class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]


class TinyPolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(TinyPolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc_mu = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc_mu(x))
        
        return x
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]


class TinyCriticNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(TinyPolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(128+act_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc_mu = nn.Linear(10, 2)

    def forward(self, x, action):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_mu(x)
        
        return x
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().numpy()[0]

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
   

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q




if __name__ == "__main__":
    print("Hello World!")
    
    