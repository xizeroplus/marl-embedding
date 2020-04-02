import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import os
from generate_dec_data_gridsmall import DecDataGenerator
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse



# parser = argparse.ArgumentParser()
# parser.add_argument("lr1", help="display a square of a given number", type=float)
# parser.add_argument("lr2", help="display a square of a given number", type=float)
# parser.add_argument("mg", help="display a square of a given number", type=int)
# parser.add_argument("lmd", help="display a square of a given number", type=float)

# args = parser.parse_args()



lr1 = 8e-2
lr2 = 1e-1
magic_number = 10
lmd = 0.8
# lr1 = args.lr1
# lr2 = args.lr2
# magic_number = args.mg
# lmd = args.lmd
sufficient_size = magic_number
batch_size = 200
sequence_size = 20
gru1_hidden_size = magic_number
fc_output_size = magic_number
reward_size = 1
filename = 'dectiger'
output_filename = str(lr1) + '-' + str(lr2) + '-' + str(magic_number) + '-' + str(lmd) + '.txt'


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GRUNet, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size


        self.gru = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True)   
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.gru(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = out.reshape(batch_size, 1, self.output_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden


class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model1, self).__init__()

        self.input_size = input_size
        self.fc = nn.Linear(input_size, fc_output_size)
        self.gru = GRUNet(fc_output_size, hidden_size, sufficient_size)
        self.tanh = nn.ReLU()
        # self.tanh = nn.ReLU()


    def forward(self, input):
        input = input.reshape(-1, self.input_size)
        input = self.fc(input)
        input = self.tanh(input)
        input = input.reshape(batch_size, 1, fc_output_size)
        
        output, hidden = self.gru(input)
        # output[batch_size][1][sufficient_size]
        return output, hidden


class Policy(nn.Module):
    def __init__(self, agent_i):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(sufficient_size, magic_number)
        # self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(magic_number, action_size[agent_i])
        self.relu = nn.ReLU()

        self.saved_log_probs = [ [] for i in range(batch_size)]   # [batch_size][sequence_size]
        self.rewards = [ [] for i in range(batch_size)]     # [batch_size][sequence_size]

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.relu(x)
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)


def get_actions(agent_i, sufficient_info, batch_size):
    # output: actions[batch_size][1]
    actions = []
    for i in range(batch_size):
        probs = policy[agent_i](sufficient_info[i].detach())     # Should be detached from the tensor graph. Otherwise you back propagate twice on a subgraph
        m = Categorical(probs)
        action = m.sample()
        policy[agent_i].saved_log_probs[i].append(m.log_prob(action))
        actions.append([action.item()])
    return actions

# def get_actions(batch_size):
#     actions = [[random.randint(0, action_size - 1)] for i in range(batch_size)]
#     # actions = [[0] for i in range(batch_size)]
#     return actions


def finish_episode():

    for agent_i in range(agent_num):

        policy_loss = [[] for i in range(batch_size)]
        returns = [[] for i in range(batch_size)]
        policy_loss_sum = 0
        for i in range(batch_size):
            R = 0
            for r in policy[agent_i].rewards[i][::-1]:
                R = r + 1 * R     # 1 was replaced by data_generator.discount
                returns[i].insert(0, R)
            returns[i] = torch.tensor(returns[i])
            returns[i] = (returns[i] - returns[i].mean()) / (returns[i].std() + eps)
            for log_prob, R in zip(policy[agent_i].saved_log_probs[i], returns[i]):
                policy_loss[i].append(-log_prob * R)
            policy_loss[i] = torch.cat(policy_loss[i]).sum()
            policy_loss_sum += policy_loss[i]

            del policy[agent_i].rewards[i][:]
            del policy[agent_i].saved_log_probs[i][:]
            
    policy_optimizer.zero_grad()
    policy_loss_sum.backward()
    policy_optimizer.step()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    # if t % 5 == 1:
    #     print(t, loss.item())


class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, sufficient_size):
        # input_size = sufficient_size + action_size
        super(Model2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sufficient_size = sufficient_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, sufficient_size * 2)

    def forward(self, sufficient_info, action):     
        # input: sufficient_info: [batch_size][sufficient_size * 2]
        # input: action: [batch_size][action_size * 2] (one-hot encoded)
        # 
        input = torch.cat((sufficient_info, action), 1)
        hidden = self.tanh(self.fc1(input))
        mean = self.fc2(hidden)
        return mean



class Model3(nn.Module):
    def __init__(self, input_size, hidden_size, reward_size, sufficient_size):
        # input_size = sufficient_size + action_size
        super(Model3, self).__init__()
        self.input_size = input_size
        self.reward_size = reward_size
        self.hidden_size = hidden_size
        self.sufficient_size = sufficient_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, reward_size)
        self.fc3 = nn.Linear(sufficient_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, sufficient_size)

    def forward(self, sufficient_info, action):     
        # input: sufficient_info: [batch_size][sufficient_size]
        # input: action: [batch_size][action_size] (one-hot encoded)
        # 
        input = torch.cat((sufficient_info, action), 1)
        hidden = self.tanh(self.fc1(input))
        reward = self.fc2(hidden)
        hidden = self.tanh(self.fc3(sufficient_info))
        mean = self.fc3(hidden)

        return reward, mean



def one_hot_encoding(xs, n):
    tmp = [[i] for i in range(n)]
    enc = OneHotEncoder(handle_unknown='ignore', categories=[[i for i in range(n)]])
    enc.fit(tmp)
    result = []
    for i in range(batch_size):
        tmpx = xs[i]
        x = [[tmpx[0]]]
        x = enc.transform(x).toarray()
        result.append(x)
    result = torch.tensor(result).float()
    # result[batch_size][1][action_size]
    return result



data_generator = DecDataGenerator(filename + '.txt')

agent_num = data_generator.agent_num
state_size = data_generator.state_size
action_size = data_generator.action_size
actions = [None for i in range(2)]
one_hot_actions = [[], []]
one_hot_last_actions = [[], []]
observation_size = data_generator.observation_size
one_hot_observations = [[], []]
sufficient_info = [None for i in range(2)]
mean_pred = [None for i in range(2)]
gaussian = [[None for i in range(batch_size)] for j in range(2)]
joint_gaussian = [None for i in range(batch_size)]
reward_pred = [[], []]
input_size = [0, 0]
action_size_sum = 0
for agent_i in range(agent_num):
    input_size[agent_i] = observation_size[agent_i] + action_size[agent_i]
    action_size_sum += action_size[agent_i]

writer = SummaryWriter()


# one_hot_actions = one_hot_encoding(actions, action_size)
# one_hot_observations = one_hot_encoding(observations, observation_size)
# input = torch.cat((one_hot_actions, one_hot_observations), 2)
# rewards = torch.tensor(rewards)


net1 = [Model1(input_size[i], gru1_hidden_size, sufficient_size) for i in range(agent_num)]
net2 = Model2(sufficient_size * 2 + action_size_sum, magic_number, sufficient_size)
net3 = [Model3(action_size_sum + sufficient_size, magic_number, reward_size, sufficient_size) for i in range(agent_num)]


policy = [Policy(i) for i in range(agent_num)]
policy_optimizer = optim.Adam(list(policy[0].parameters()) + list(policy[1].parameters()), lr=lr1, betas=(0.95, 0.95))
eps = np.finfo(np.float32).eps.item()


# criterion = nn.MSELoss(reduction='sum')
criterion = nn.SmoothL1Loss(reduction='sum')
optimizer = optim.Adam(list(net1[0].parameters()) + list(net3[0].parameters()) + list(net2.parameters()) + list(net1[1].parameters()) + list(net3[1].parameters()), lr=lr2, betas=(0.95, 0.95))



for t in range(30):
    # output_file = open(output_filename, 'a')
    reward_sum = 0
    loss = 0
    data_generator.init_environment(batch_size)
    last_actions = [[[0] for i in range(batch_size)] for j in range(agent_num)]
    for agent_i in range(agent_num):
        one_hot_last_actions[agent_i] = one_hot_encoding(last_actions[agent_i], action_size[agent_i])#.detach()
        # gaussian = [MultivariateNormal(torch.zeros(sufficient_size), torch.eye(sufficient_size)) for j in range(batch_size)]

    for sequence_i in range(sequence_size):
        observations0, observations1, rewards = data_generator.interact_with_environment(last_actions[0], last_actions[1])
        observations = [observations0, observations1]
        rewards = torch.tensor(rewards)
        if sequence_i != 0:
            reward_sum += rewards.sum().item()
        for agent_i in range(agent_num):
            one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size[agent_i])
            input = torch.cat((one_hot_last_actions[agent_i], one_hot_observations[agent_i]), 2)
            for i in range(batch_size):
                policy[agent_i].rewards[i].append(rewards[i][0])
            if sequence_i != 0:
                loss += lmd * criterion(reward_pred[agent_i], rewards)

            sufficient_info[agent_i], hidden = net1[agent_i](input)

            # Forcing the first two dimensions of information state to be a belief over nature state
            # for i in range(batch_size):
            #     current_state = data_generator.current_states[i]
            #     loss += -0.1 * torch.log(sufficient_info[agent_i][i][0][current_state])

            actions[agent_i] = get_actions(agent_i, sufficient_info[agent_i], batch_size)
            one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size[agent_i])
    
        for agent_i in range(agent_num):
            input = torch.cat((one_hot_actions[agent_i][:, 0, :], one_hot_actions[1 - agent_i][:, 0, :]), 1)
            reward_pred[agent_i], mean_pred[agent_i] = net3[agent_i](sufficient_info[agent_i][:, 0, :], input)
            for i in range(batch_size):
                gaussian[agent_i][i] = MultivariateNormal(mean_pred[agent_i][i], torch.eye(sufficient_size))   

        for agent_i in range(agent_num):
            for i in range(batch_size):
                # condition iv
                loss += (1 - lmd) * (-gaussian[agent_i][i].log_prob(sufficient_info[1 - agent_i][i, 0, :]))

            last_actions[agent_i] = actions[agent_i]
            one_hot_last_actions[agent_i] = one_hot_actions[agent_i]

        if sequence_i != 0:
            for i in range(batch_size):
                # condition ii
                loss += (1 - lmd) * (-joint_gaussian[i].log_prob(torch.cat((sufficient_info[0][i, 0, :], sufficient_info[1][i, 0, :]), 0))) 

        joint_mean_pred = net2(torch.cat((one_hot_actions[0][:, 0, :], one_hot_actions[1][:, 0, :]), 1), torch.cat((sufficient_info[0][:, 0, :], sufficient_info[1][:, 0, :]), 1))
        for i in range(batch_size):
            joint_gaussian[i] = MultivariateNormal(joint_mean_pred[i], torch.eye(sufficient_size * 2))


    finish_episode()

    writer.add_scalar(filename + '/reward', reward_sum / batch_size / data_generator.discount, t)
    print(t, reward_sum / data_generator.discount / batch_size)
    # output_file.write(str(t) + ' ' + str(reward_sum / data_generator.discount / batch_size) + '\n')
    reward_sum = 0

    # output_file.close()

