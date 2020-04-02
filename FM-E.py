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
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--sequence_size', help="length of the horizon", type=int, default=20)
parser.add_argument('--length', help="length of the fixed memory", type=int, default=4)
parser.add_argument('--lr', help="learning rate", type=float, default=1e-2)
parser.add_argument('--task', help="task problem", default='boxpushing')
args = parser.parse_args()

SEQUENCE_SIZE = args.sequence_size
LENGTH = args.length
lr = args.lr
task = args.task
BATCH_SIZE = 400
MEMORY_SIZE = 4000
HIDDEN_SIZE = 10
TOTAL_EPISODES = 40000
TEST_TOTAL_EPISODES = 2000
TARGET_UPDATE_FREQ = 100 * SEQUENCE_SIZE
BACK_PROP_FREQ = 1 * SEQUENCE_SIZE
INITIAL_EPSILON = 0.8
FINAL_EPSILON = 0.0


if task == 'boxpushing':
    from env.boxpushing import Environment
elif task == 'grid3x3':
    from env.grid3x3 import Environment
else:
    from env.dectiger import Environment


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


def train():

    # Initialize experience memory
    for episode in range(0, MEMORY_SIZE // SEQUENCE_SIZE // 2 + 1):
        env.init_environment(batch_size=1)
        embedding = [deque([0.0 for i in range(LENGTH * (action_size + observation_size + 1))], maxlen=LENGTH * (action_size + observation_size + 1)) for i in range(2)]
        
        for _ in range(SEQUENCE_SIZE):
            actions = [[random.randint(0, action_size - 1)] for i in range(2)]
            observations[0], observations[1], rewards = env.step(actions[0], actions[1])

            for agent_i in range(agent_num):
                one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
                one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
                tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
                current_state = torch.FloatTensor(embedding[agent_i])
                embedding[agent_i].extend(torch.cat((tmp, one_hot_observations[agent_i][0]), 0).tolist())
                new_state = torch.FloatTensor(embedding[agent_i])
                memory.add_episode((current_state, actions[agent_i][0], rewards[0], new_state))
        

    epsilon = INITIAL_EPSILON
    reward_stat = []
    total_steps = 0
    total_reward = 0
    total_loss = 0


    # Start training
    for episode in range(TOTAL_EPISODES):
        env.init_environment(batch_size=1)
        embedding = [deque([0.0 for i in range(LENGTH * (action_size + observation_size + 1))], maxlen=LENGTH * (action_size + observation_size + 1)) for i in range(2)]

        for step in range(SEQUENCE_SIZE):
            total_steps += 1
            current_state = [None for i in range(2)]
            for agent_i in range(2):
                current_state[agent_i] = torch.FloatTensor(embedding[agent_i])
            if np.random.rand(1) < epsilon:
                actions = [[random.randint(0, action_size - 1)] for i in range(2)]
            else:
                for agent_i in range(2):
                    q_values = main_model(current_state[agent_i], batch_size=1)[0]
                    actions[agent_i] = [int(torch.argmax(q_values))]

            observations[0], observations[1], rewards = env.step(actions[0], actions[1])
            total_reward += rewards[0]
            for agent_i in range(2):
                one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
                one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)
                tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
                embedding[agent_i].extend(torch.cat((tmp, one_hot_observations[agent_i][0]), 0).tolist())
                new_state = torch.FloatTensor(embedding[agent_i])
                memory.add_episode((current_state[agent_i], actions[agent_i][0], rewards[0], new_state))


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
            writer.add_scalar(task + '/reward', total_reward / 100, episode)
            writer.add_scalar(task + '/loss', total_loss.item() / 100, episode)
            total_reward = 0
            total_loss = 0

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (TOTAL_EPISODES - 2000)



def test():
    result = 0
    total_reward = 0
    for length in [3, 4, 5, 6, 7]:
        TEST_SEQUENCE_SIZE = length

        for episode in range(TEST_TOTAL_EPISODES):
            env.init_environment(batch_size=1)
            embedding = [deque([0.0 for i in range(LENGTH * (action_size + observation_size + 1))], maxlen=LENGTH * (action_size + observation_size + 1)) for i in range(2)]
            current_discount = 1.0

            for _ in range(TEST_SEQUENCE_SIZE):
                current_state = [None for i in range(2)]
                actions = [None for i in range(2)]
                for agent_i in range(2):
                    current_state[agent_i] = torch.FloatTensor(embedding[agent_i])
                for agent_i in range(2):
                    q_values = main_model(current_state[agent_i], batch_size=1)[0]
                    actions[agent_i] = [int(torch.argmax(q_values))]

                observations[0], observations[1], rewards = env.step(actions[0], actions[1])
                total_reward += rewards[0] * current_discount
                result += rewards[0] * current_discount
                for agent_i in range(2):
                    one_hot_actions[agent_i] = one_hot_encoding(actions[agent_i], action_size)
                    one_hot_observations[agent_i] = one_hot_encoding(observations[agent_i], observation_size)       
                    tmp = torch.cat((torch.FloatTensor([agent_i]), one_hot_actions[agent_i][0]), 0)
                    embedding[agent_i].extend(torch.cat((tmp, one_hot_observations[agent_i][0]), 0).tolist())
                    new_state = torch.FloatTensor(embedding[agent_i])

            if episode % 100 == 99:
                print(episode, total_reward / 100)
                total_reward = 0

        file = open('FM-E_' + task + '.txt', 'a')
        file.write(str(TEST_SEQUENCE_SIZE) + '\t' + str(result / TEST_TOTAL_EPISODES) + '\n')
        file.close()
        result = 0



if __name__ == "__main__":
    
    env = Environment(task + '.txt')

    agent_num = env.agent_num
    action_size = env.action_size[0]
    observation_size = env.observation_size[0]

    writer = SummaryWriter()

    memory = Memory(memory_size=MEMORY_SIZE)
    main_model = Model(input_size=LENGTH * (action_size + observation_size + 1), hidden_size=HIDDEN_SIZE, output_size=action_size).float()
    target_model = Model(input_size=LENGTH * (action_size + observation_size + 1), hidden_size=HIDDEN_SIZE, output_size=action_size).float()

    target_model.load_state_dict(main_model.state_dict())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(main_model.parameters(), lr=lr)

    one_hot_actions = [None for i in range(2)]
    observations = [None for i in range(2)]
    one_hot_observations = [None for i in range(2)]

    train()

    test()

