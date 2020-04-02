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
import sklearn.decomposition


BATCH_SIZE = 400
MEMORY_SIZE = 4000
HIDDEN_SIZE = 6
SEQUENCE_SIZE = 20 # should be 5
TOTAL_EPISODES = 40000
TEST_TOTAL_EPISODES = 2000
TARGET_UPDATE_FREQ = 100 * SEQUENCE_SIZE
BACK_PROP_FREQ = 1 * SEQUENCE_SIZE      # this should be much smaller, e.g., 5
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.0
PCA_LENGTH = 8
lr = 1e-2
filename = 'dectiger'




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
main_model = Model(input_size=PCA_LENGTH + 1, hidden_size=HIDDEN_SIZE, output_size=action_size).float()
target_model = Model(input_size=PCA_LENGTH + 1, hidden_size=HIDDEN_SIZE, output_size=action_size).float()

target_model.load_state_dict(main_model.state_dict())
# criterion = nn.SmoothL1Loss()     #SmoothL1Loss does not work
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(main_model.parameters(), lr=lr)

one_hot_actions = [None for i in range(2)]
observations = [None for i in range(2)]
one_hot_observations = [None for i in range(2)]


# X = []
# current_sample = [0 for _ in range((action_size + observation_size) * (SEQUENCE_SIZE))]
# def add_sample(depth, length):
#     if depth >= length:
#         X.append(current_sample.copy())
#         return
#     for a in range(action_size):
#         for o in range(observation_size):
#             one_hot_actions = [[0 for _ in range(action_size)]]
#             one_hot_observations = [[0 for _ in range(observation_size)]]
#             one_hot_actions[0][a] = 1
#             one_hot_observations[0][o] = 1
#             # one_hot_actions = one_hot_encoding([a], action_size)
#             # one_hot_observations = one_hot_encoding([o], observation_size)
#             for i in range((action_size + observation_size) * (SEQUENCE_SIZE - length + depth), (action_size + observation_size) * (SEQUENCE_SIZE - length + depth + 1) - observation_size, 1):
#                 current_sample[i] = one_hot_actions[0][i - (action_size + observation_size) * (SEQUENCE_SIZE - length + depth)]
#             for i in range((action_size + observation_size) * (SEQUENCE_SIZE - length + depth + 1) - observation_size, (action_size + observation_size) * (SEQUENCE_SIZE - length + depth + 1), 1):
#                 current_sample[i] = one_hot_observations[0][i - (action_size + observation_size) * (SEQUENCE_SIZE - length + depth + 1) + observation_size]
#             add_sample(depth + 1, length)


# for length in range(SEQUENCE_SIZE + 1):
#     add_sample(0, length)
# print(len(X))

X = []
current_sample = [0 for _ in range((action_size + observation_size) * (SEQUENCE_SIZE))]
X.append(current_sample)
sample_size = 300000

for _ in range(sample_size):
    current_sample = [0 for _ in range((action_size + observation_size) * (SEQUENCE_SIZE))]
    length = random.randint(1, SEQUENCE_SIZE)
    for i in range(length):
        a = random.randint(0, action_size - 1)
        o = random.randint(0, observation_size - 1)
        current_sample[(SEQUENCE_SIZE - length + i) * (action_size + observation_size) + a] = 1
        current_sample[(SEQUENCE_SIZE - length + i) * (action_size + observation_size) + action_size + o] = 1
        X.append(current_sample)



pca = sklearn.decomposition.PCA(n_components=PCA_LENGTH)
pca.fit(np.array(X))
print('pca done')


# Initialize experience memory
for episode in range(0, MEMORY_SIZE // SEQUENCE_SIZE // 2 + 1):
    data_generator.init_environment(batch_size=1)
    embedding = [deque([0.0 for i in range((SEQUENCE_SIZE) * (action_size + observation_size))], maxlen=(SEQUENCE_SIZE) * (action_size + observation_size)) for i in range(2)]
    
    for step in range(SEQUENCE_SIZE):
        actions = [[random.randint(0, action_size - 1)] for i in range(2)]
        observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])

        for agent_i in range(agent_num):    
            one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
            one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
            # TODO here

            current_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
            current_state = torch.cat((torch.FloatTensor([agent_i]), current_state), 0)

            embedding[agent_i].extend(torch.cat((one_hot_actions[agent_i][0], one_hot_observations[agent_i][0]), 0).tolist())
            new_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
            new_state = torch.cat((torch.FloatTensor([agent_i]), new_state), 0)
            memory.add_episode((current_state, actions[agent_i][0], rewards[0], new_state))



epsilon = INITIAL_EPSILON
reward_stat = []
total_steps = 0
total_reward = 0
total_loss = 0


# Start training
for episode in range(TOTAL_EPISODES):
    data_generator.init_environment(batch_size=1)
    embedding = [deque([0.0 for i in range((SEQUENCE_SIZE) * (action_size + observation_size))], maxlen=(SEQUENCE_SIZE) * (action_size + observation_size)) for i in range(2)]

    for step in range(SEQUENCE_SIZE):
        total_steps += 1
        current_state = [None for i in range(2)]
        for agent_i in range(2):
            current_state[agent_i] = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
            current_state[agent_i] = torch.cat((torch.FloatTensor([agent_i]), current_state[agent_i]), 0)
        if np.random.rand(1) < epsilon:
            actions = [[random.randint(0, action_size - 1)] for i in range(2)]
        else:
            for agent_i in range(2):
                q_values = main_model(current_state[agent_i], batch_size=1)[0]
                actions[agent_i] = [int(torch.argmax(q_values))]

        observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])
        total_reward += rewards[0]
        if episode % 100 == 0 and episode >= TOTAL_EPISODES - 2000:
            print(actions[0][0], actions[1][0], observations[0][0], observations[1][0], rewards[0])
        
        for agent_i in range(agent_num):    
            one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
            one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)

            current_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
            current_state = torch.cat((torch.FloatTensor([agent_i]), current_state), 0)

            embedding[agent_i].extend(torch.cat((one_hot_actions[agent_i][0], one_hot_observations[agent_i][0]), 0).tolist())
            new_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
            new_state = torch.cat((torch.FloatTensor([agent_i]), new_state), 0)
            memory.add_episode((current_state, actions[agent_i][0], rewards[0], new_state))


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
                loss.backward()
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
    # if episode > TOTAL_EPISODES - 2000:
    #     epsilon = 0.0
    
    



# test
report = 0.0
result = 0
for length in [3, 4, 5, 6, 7, SEQUENCE_SIZE]:
    TEST_SEQUENCE_SIZE = length

    for episode in range(TEST_TOTAL_EPISODES):
        data_generator.init_environment(batch_size=1)
        embedding = [deque([0.0 for i in range((SEQUENCE_SIZE) * (action_size + observation_size))], maxlen=(SEQUENCE_SIZE) * (action_size + observation_size)) for i in range(2)]
        current_discount = 1.0

        for step in range(TEST_SEQUENCE_SIZE):
            total_steps += 1
            current_state = [None for i in range(2)]
            for agent_i in range(2):
                current_state[agent_i] = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
                current_state[agent_i] = torch.cat((torch.FloatTensor([agent_i]), current_state[agent_i]), 0)
            for agent_i in range(2):
                q_values = main_model(current_state[agent_i], batch_size=1)[0]
                actions[agent_i] = [int(torch.argmax(q_values))]

            observations[0], observations[1], rewards = data_generator.interact_with_environment(actions[0], actions[1])
            total_reward += rewards[0] * current_discount
            result += rewards[0] * current_discount
            # current_discount *= data_generator.discount
            if episode % 100 == 0:
                print(actions[0][0], actions[1][0], observations[0][0], observations[1][0], rewards[0])
            for agent_i in range(agent_num):    
                one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
                one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)

                current_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
                current_state = torch.cat((torch.FloatTensor([agent_i]), current_state), 0)

                embedding[agent_i].extend(torch.cat((one_hot_actions[agent_i][0], one_hot_observations[agent_i][0]), 0).tolist())
                new_state = torch.FloatTensor(pca.transform(np.array([embedding[agent_i]]))[0])
                new_state = torch.cat((torch.FloatTensor([agent_i]), new_state), 0)


        if episode % 100 == 99:
            print(episode, total_reward / 100)
            writer.add_scalar(filename + '/reward', total_reward / 100, episode)
            total_reward = 0
            total_loss = 0

    print(result / TEST_TOTAL_EPISODES)
    file = open('pca_dectiger.txt', 'a')
    file.write(str(TEST_SEQUENCE_SIZE) + '\t' + str(result / TEST_TOTAL_EPISODES) + '\n')
    file.close()
    report += result / TEST_TOTAL_EPISODES
    result = 0


# main_model = torch.load('main_model.pkl')

os._exit(0)

