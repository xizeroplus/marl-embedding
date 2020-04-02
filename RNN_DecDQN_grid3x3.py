import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque 
import torch.optim as optim
import os
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal


BATCH_SIZE = 400
MEMORY_SIZE = 4000
HIDDEN_SIZE = 10
SEQUENCE_SIZE = 10 # should be 5
TOTAL_EPISODES = 40000
TEST_TOTAL_EPISODES = 2000
TARGET_UPDATE_FREQ = 100 * SEQUENCE_SIZE
BACK_PROP_FREQ = 1 * SEQUENCE_SIZE      # this should be much smaller, e.g., 5
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0
LENGTH = 7
lr = 1e-2
filename = 'grid3x3'




if filename == 'boxpushing':
    from generate_dec_data_boxpushing import DecDataGenerator
elif filename == 'grid3x3':
    from generate_dec_data_grid3x3 import DecDataGenerator
else:
    from generate_dec_data import DecDataGenerator


class Memory():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
    
    def add_episode(self, epsiode):
        self.memory.append(epsiode)
    
    def get_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, input, batch_size):
        input = torch.FloatTensor(input).view(batch_size, -1)
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        
        return output


def one_hot_encoding(xs, n):
    # xs[batch_size]
    tmp = [[i] for i in range(n)]
    enc = OneHotEncoder(handle_unknown='ignore', categories=[[i for i in range(n)]])
    enc.fit(tmp)
    xs = np.expand_dims(np.array(xs), axis=1)
    result = enc.transform(xs).toarray()
    result = torch.tensor(result).float()
    # result[batch_size][action_size]
    return result




data_generator = DecDataGenerator(filename + '.txt')

agent_num = data_generator.agent_num
state_size = data_generator.state_size
action_size = data_generator.action_size[0]
observation_size = data_generator.observation_size[0]
input_size = observation_size + action_size

writer = SummaryWriter()

memory = Memory(memory_size=MEMORY_SIZE)
main_model = Model(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, output_size=action_size).float()
target_model = Model(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, output_size=action_size).float()
lstm = nn.LSTM(action_size + observation_size + 1, HIDDEN_SIZE, batch_first=True)

target_model.load_state_dict(main_model.state_dict())
# criterion = nn.SmoothL1Loss()     #SmoothL1Loss does not work
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(main_model.parameters()), lr=lr)
# optimizer_dqn = torch.optim.Adam(list(main_model.parameters()), lr=lr)

one_hot_actions = [None for i in range(2)]
observations = [None for i in range(2)]
one_hot_observations = [None for i in range(2)]

# Initialize experience memory
for episode in range(0, MEMORY_SIZE // SEQUENCE_SIZE // 2 + 1):
    data_generator.init_environment(batch_size=1)
    hidden = [(torch.randn(1, 1, HIDDEN_SIZE).detach(), torch.randn(1, 1, HIDDEN_SIZE).detach()) for i in range(2)]
    current_state = [torch.zeros(1, 1, HIDDEN_SIZE) for i in range(2)]
    new_state = [None for i in range(2)]
    # embedding = [deque([0.0 for i in range(LENGTH * (action_size + observation_size))], maxlen=LENGTH * (action_size + observation_size)) for i in range(2)]
    
    for step in range(SEQUENCE_SIZE):
        actions = [[random.randint(0, action_size - 1)] for i in range(2)]
        observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])

        for agent_i in range(agent_num):
            one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
            one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
            tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
            input = torch.cat((tmp, one_hot_observations[agent_i][0]), dim=0)
            
            new_state[agent_i], hidden[agent_i] = lstm(input.view(1, 1, -1), hidden[agent_i])
            hidden[agent_i] = (hidden[agent_i][0].detach(), hidden[agent_i][1].detach())
            
            memory.add_episode((current_state[agent_i][0][0], actions[agent_i][0], rewards[0], new_state[agent_i][0][0]))
            current_state[agent_i] = new_state[agent_i]
    
    

epsilon = INITIAL_EPSILON
reward_stat = []
total_steps = 0
total_reward = 0
total_loss = 0


# Start training
for episode in range(TOTAL_EPISODES):
    data_generator.init_environment(batch_size=1)
    hidden = [(torch.randn(1, 1, HIDDEN_SIZE).detach(), torch.randn(1, 1, HIDDEN_SIZE).detach()) for i in range(2)]
    current_state = [torch.zeros(1, 1, HIDDEN_SIZE) for i in range(2)]

    for step in range(SEQUENCE_SIZE):
        total_steps += 1
        if np.random.rand(1) < epsilon:
            actions = [[random.randint(0, action_size - 1)] for i in range(2)]
        else:
            for agent_i in range(2):
                q_values = main_model(current_state[agent_i], batch_size=1)[0]
                actions[agent_i] = [int(torch.argmax(q_values))]
        
        # id = episode // 2000 % 2
        # q_values = main_model(current_state[id], batch_size=1)[0]
        # actions[id] = [int(torch.argmax(q_values))]
        # if np.random.rand(1) < epsilon:
        #     actions[1 - id] = [random.randint(0, action_size - 1)]
        # else:
        #     q_values = main_model(current_state[1 - id], batch_size=1)[0]
        #     actions[1 - id] = [int(torch.argmax(q_values))]

        observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])
        total_reward += rewards[0]
        for agent_i in range(2):
            one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
            one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
            tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
            input = torch.cat((tmp, one_hot_observations[agent_i][0]), dim=0)
            
            new_state[agent_i], hidden[agent_i] = lstm(input.view(1, 1, -1), hidden[agent_i])
            hidden[agent_i] = (hidden[agent_i][0].detach(), hidden[agent_i][1].detach())
            
            
            memory.add_episode((current_state[agent_i][0][0], actions[agent_i][0], rewards[0], new_state[agent_i][0][0]))
            current_state[agent_i] = new_state[agent_i]
            


        if (total_steps % TARGET_UPDATE_FREQ) == 0:
            target_model.load_state_dict(main_model.state_dict())
       
        if (total_steps % BACK_PROP_FREQ) == 0:     

            batch = memory.get_batch(batch_size=BATCH_SIZE)
            
            current_states = []
            local_actions = []
            local_rewards = []
            next_states = []

            for sample in batch:
                current_states.append(sample[0])
                local_actions.append(sample[1])
                local_rewards.append(sample[2])
                next_states.append(sample[3])
            
            current_states = torch.cat(current_states, dim=0)   # [batch_size][embedding_size]
            local_actions = torch.LongTensor(local_actions)
            local_rewards = torch.FloatTensor(local_rewards)    # [batch_size]
            next_states = torch.cat(next_states, dim=0)

            next_q_values = target_model(next_states, batch_size=BATCH_SIZE)    # [batch_size][action_size]
            next_q_max_value, _ = next_q_values.detach().max(dim=1)    # [batch_size]
            target_values = local_rewards + 0.75 * next_q_max_value    # There should be a gamma factor
            
            q_values = main_model(current_states, batch_size=BATCH_SIZE)     # [batch_size][action_size]
            current_values = torch.gather(q_values, dim=1, index=local_actions.unsqueeze(dim=1)).squeeze(dim=1)
            
            loss = criterion(current_values, target_values)
            total_loss += loss
            
            
            if episode <= TOTAL_EPISODES - 2000:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


    reward_stat.append(total_reward)
    if episode % 100 == 99:
        print(episode, total_reward / 100, total_loss.item() / 100)
        writer.add_scalar(filename + '/reward', total_reward / 100, episode)
        writer.add_scalar(filename + '/loss', total_loss.item() / 100, episode)
        total_reward = 0
        total_loss = 0

    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (TOTAL_EPISODES - 4000)



    



# test
result = 0
for length in [6, 7, 8, 9, 10, SEQUENCE_SIZE]:
    TEST_SEQUENCE_SIZE = length

    for episode in range(TEST_TOTAL_EPISODES):
        data_generator.init_environment(batch_size=1)
        hidden = [(torch.randn(1, 1, HIDDEN_SIZE).detach(), torch.randn(1, 1, HIDDEN_SIZE).detach()) for i in range(2)]
        current_state = [torch.zeros(1, 1, HIDDEN_SIZE) for i in range(2)]
        current_discount = 1.0

        for step in range(TEST_SEQUENCE_SIZE):
            total_steps += 1
            for agent_i in range(2):
                q_values = main_model(current_state[agent_i], batch_size=1)[0]
                actions[agent_i] = [int(torch.argmax(q_values))]

            observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])
            total_reward += rewards[0] * current_discount
            result += rewards[0] * current_discount
            # current_discount *= data_generator.discount
            if episode % 100 == 0:
                print(actions[0][0], actions[1][0], observations[0][0], observations[1][0], rewards[0])
            for agent_i in range(2):
                one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
                one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
                tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
                input = torch.cat((tmp, one_hot_observations[agent_i][0]), dim=0)
            
                new_state[agent_i], hidden[agent_i] = lstm(input.view(1, 1, -1), hidden[agent_i])
                hidden[agent_i] = (hidden[agent_i][0].detach(), hidden[agent_i][1].detach())
            
                current_state[agent_i] = new_state[agent_i]


        if episode % 100 == 99:
            print(episode, total_reward / 100)
            writer.add_scalar(filename + '/reward', total_reward / 100, episode)
            total_reward = 0
            total_loss = 0

    print(result / TEST_TOTAL_EPISODES)
    file = open('RNN_grid3x3.txt', 'a')
    file.write(str(TEST_SEQUENCE_SIZE) + '\t' + str(result / TEST_TOTAL_EPISODES) + '\n')
    file.close()
    result = 0


# main_model = torch.load('main_model.pkl')
