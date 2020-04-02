import os
import random
import numpy as np

class DecDataGenerator:


    def read_matrix(self, n, m):
        line = self.f.readline()
        while not len(line) or line.startswith('#'):
            line = self.f.readline()
        line = line.strip()
        if line == 'identity':
            result = [[0.0 for i in range(n)] for j in range(m)]
            for i in range(n):
                result[i][i] = 1.0
            return result
        elif line == 'uniform':
            result = [[1.0 / m for i in range(n)] for j in range(m)]
            return result
        items = line.split(' ')
        result = [[0.0 for i in range(n)] for j in range(m)]
        for i in range(n):
            for j in range(m):
                result[i][j] = float(items[j])
            line = self.f.readline().strip()
            items = line.split(' ')
        return result

    def __init__(self, filename):

        self.agent_num = 2
        self.discount = 1
        self.action_size = []
        self.state_size = 0
        self.observation_size = []
        self.reward_flag = 1.0
        self.readline_count = 0

        self.state_names = []
        self.state_dict = {}
        self.action_names = []
        self.action_dict = []
        self.observation_names = []
        self.observation_dict = []

        self.has_initial = True
        self.initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # We need to specify the initial belief of each problem manually
        self.T = []  # T[s][a_0][a_1][s']
        self.O = []  # O[s'][a_0][a_1][o_0][o_1]
        self.R = []  # R[s][a_0][a_1][s'][o_0][o_1]

        self.current_states = []
        self.current_discount = 1.0
        self.current_batch_size = 1


        self.f = open(filename, 'r')
        while True:
            line = self.f.readline()
            if not line:
                break
            line = line.strip()
            if not len(line) or line.startswith('#'):
                continue
            self.readline_count += 1
            if self.readline_count <= 7:
                items = line.split(' ')
                if items[0] == 'agents:':
                    self.agent_num = int(items[1])
                    self.action_names = [[] for _ in range(self.agent_num)]
                    # self.action_dict = [{} for _ in range(self.agent_num)]
                    self.observation_names = [[] for _ in range(self.agent_num)]
                    # self.observation_dict = [{} for _ in range(self.agent_num)]
                    self.action_size = [0 for _ in range(self.agent_num)]
                    self.observation_size = [0 for _ in range(self.agent_num)]
                elif items[0] == 'discount:':
                    self.discount = float(items[1])
                elif items[0] == 'values:':
                    if items[1] == 'cost':
                        self.reward_flag = -1.0
                elif items[0] == 'states:':
                    if len(items) == 2 and items[1].isdigit():
                        self.state_size = int(items[1])
                        for i in range(self.state_size):
                            self.state_names.append(str(i))
                            self.state_dict[str(i)] = i
                    else:
                        self.state_size = len(items) - 1
                        for i in range(1, len(items)):
                            self.state_names.append(items[i])
                            self.state_dict[items[i]] = i - 1
                elif items[0] == 'actions:':
                    for agent in range(self.agent_num):
                        line = self.f.readline()
                        while not len(line) or line.startswith('#'):
                            line = self.f.readline()
                        items = line.strip().split(' ')
                        if len(items) == 1 and items[0].isdigit():
                            self.action_size[agent] = int(items[0])
                        else:
                            self.action_size[agent] = len(items)
                            tmp_dict = {}
                            for i in range(0, len(items)):
                                self.action_names[agent].append(items[i])
                                tmp_dict[items[i]] = i
                            self.action_dict.append(tmp_dict)
                elif items[0] == 'observations:':
                    for agent in range(self.agent_num):
                        line = self.f.readline()
                        while not len(line) or line.startswith('#'):
                            line = self.f.readline()
                        items = line.strip().split(' ')
                        if len(items) == 1 and items[0].isdigit():
                            self.observation_size[agent] = int(items[0])
                        else:
                            self.observation_size[agent] = len(items)
                            tmp_dict = {}
                            for i in range(0, len(items)):
                                self.observation_names[agent].append(items[i])
                                tmp_dict[items[i]] = i
                            self.observation_dict.append(tmp_dict)
                else:
                    items = line.strip().split(':')
                    if items[0] == 'start':
                        if filename == 'dectiger.txt':
                            self.has_initial = False
                            line = self.f.readline()
                        else:
                            self.has_initial = True
                            line = self.f.readline()
                            pass
                            # while not len(line) or line.startswith('#'):
                            #     line = self.f.readline()
                            # items = line.strip().split(' ')
                            # for i in range(self.state_size):
                            #     self.initial.append(float(items[i]))
                            # self.initial = np.array(self.initial)
                            # self.initial = list(self.initial / self.initial.sum())
                            
                            # print(len(self.initial))
                    else:
                        print("input format not supported")
                if self.readline_count == 7:
                    self.T = [[[[0.0 for i in range(self.state_size)] for j in range(self.action_size[1])] for _ in range(self.action_size[0])] for k in range(self.state_size)]
                    self.O = [[[[[0.0 for __ in range(self.observation_size[1])] for i in range(self.observation_size[0])] for k in range(self.action_size[1])] for _ in range(self.action_size[0])] for j in range(self.state_size)]
                    self.R = [[[[[[0.0 for __ in range(self.observation_size[1])] for l in range(self.observation_size[0])] for i in range(self.state_size)] for j in range(self.action_size[1])] for _ in range(self.action_size[0])] for k in range(self.state_size)]
            else:
                items = line.strip().split(':')
                if items[0] == 'T': 
                    lower = []
                    upper = []
                    if len(items) <= 3:
                        items[1] = items[1].strip()
                        if items[1] == '*':
                            lower = [0, 0]
                            upper = [self.action_size[0], self.action_size[1]]
                        else:
                            lower = [self.action_dict[_][items[1].split(' ')[_]] for _ in range(2)]
                            upper = [self.action_dict[_][items[1].split(' ')[_]] + 1 for _ in range(2)]

                        tmp = self.read_matrix(self.state_size, self.state_size)
                        for a1 in range(lower[0], upper[0]):
                            for a2 in range(lower[1], upper[1]):
                                for s1 in range(self.state_size):
                                    for s2 in range(self.state_size):
                                        self.T[s1][a1][a2][s2] = tmp[s1][s2]
                    else:
                        lower = []
                        upper = []
                        a1 = items[1].strip().split(' ')[0]
                        a2 = items[1].strip().split(' ')[1]
                        if a1 == '*':
                            lower.append(0)
                            upper.append(self.action_size[0])
                        else:
                            lower.append(int(a1))
                            upper.append(int(a1) + 1)

                        if a2 == '*':
                            lower.append(0)
                            upper.append(self.action_size[1])
                        else:
                            lower.append(int(a2))
                            upper.append(int(a2) + 1)

                        s1 = items[2].strip()
                        s2 = items[3].strip()

                        if s1 == '*':
                            lower.append(0)
                            upper.append(self.state_size)
                        else:
                            lower.append(int(s1))
                            upper.append(int(s1) + 1)

                        if s2 == '*':
                            lower.append(0)
                            upper.append(self.state_size)
                        else:
                            lower.append(int(s2))
                            upper.append(int(s2) + 1)


                        for a1 in range(lower[0], upper[0]):
                            for a2 in range(lower[1], upper[1]):
                                for s1 in range(lower[2], upper[2]):
                                    for s2 in range(lower[3], upper[3]):
                                        self.T[s1][a1][a2][s2] = float(items[4])


                elif items[0] == 'O':

                    lower = []
                    upper = []
                    if len(items) <= 3:
                        if filename == 'dectiger.txt':
                            for a1 in range(3):
                                for a2 in range(3):
                                    for s1 in range(self.state_size):
                                        for o1 in range(2):
                                            for o2 in range(2):
                                                self.O[s1][a1][a2][o1][o2] = 0.25
                            line = self.f.readline()
                        else:
                            print('Observation format not supported')
                    else:
                        lower = []
                        upper = []
                        if items[1].strip() == '*':
                            a1 = '*'
                            a2 = '*'
                        else:
                            a1 = items[1].strip().split(' ')[0]
                            a2 = items[1].strip().split(' ')[1]
                        if a1 == '*':
                            lower.append(0)
                            upper.append(self.action_size[0])
                        else:
                            lower.append(int(a1))
                            upper.append(int(a1) + 1)

                        if a2 == '*':
                            lower.append(0)
                            upper.append(self.action_size[1])
                        else:
                            lower.append(int(a2))
                            upper.append(int(a2) + 1)

                        s1 = items[2].strip()

                        if s1 == '*':
                            lower.append(0)
                            upper.append(self.state_size)
                        else:
                            lower.append(int(s1))
                            upper.append(int(s1) + 1)

                        o1 = items[3].strip().split(' ')[0]
                        o2 = items[3].strip().split(' ')[1]
                        if o1 == '*':
                            lower.append(0)
                            upper.append(self.observation_size[0])
                        else:
                            lower.append(int(o1))
                            upper.append(int(o1) + 1)

                        if o2 == '*':
                            lower.append(0)
                            upper.append(self.observation_size[1])
                        else:
                            lower.append(int(o2))
                            upper.append(int(o2) + 1)


                        for a1 in range(lower[0], upper[0]):
                            for a2 in range(lower[1], upper[1]):
                                for s1 in range(lower[2], upper[2]):
                                    for o1 in range(lower[3], upper[3]):
                                        for o2 in range(lower[4], upper[4]):
                                            self.O[s1][a1][a2][o1][o2] = float(items[4])

                elif items[0] == 'R':
                    
                    lower = []
                    upper = []
                    if len(items) <= 3:
                        if filename == 'dectiger.txt':
                            print('Observation format not supported')
                        else:
                            print('Observation format not supported')
                    else:
                        lower = []
                        upper = []
                        if items[1].strip() == '*':
                            a1 = '*'
                            a2 = '*'
                        else:
                            a1 = items[1].strip().split(' ')[0]
                            a2 = items[1].strip().split(' ')[1]
                        if a1 == '*':
                            lower.append(0)
                            upper.append(self.action_size[0])
                        else:
                            lower.append(int(a1))
                            upper.append(int(a1) + 1)

                        if a2 == '*':
                            lower.append(0)
                            upper.append(self.action_size[1])
                        else:
                            lower.append(int(a2))
                            upper.append(int(a2) + 1)

                        s1 = items[2].strip()
                        s2 = items[3].strip()

                        if s1 == '*':
                            lower.append(0)
                            upper.append(self.state_size)
                        else:
                            lower.append(int(s1))
                            upper.append(int(s1) + 1)

                        if s2 == '*':
                            lower.append(0)
                            upper.append(self.state_size)
                        else:
                            lower.append(int(s2))
                            upper.append(int(s2) + 1)

                        # o1 = items[4].strip().split(' ')[0]
                        # o2 = items[4].strip().split(' ')[1]
                        o1 = o2 = '*'   # in most cases rewards do not depend on observations

                        if o1 == '*':
                            lower.append(0)
                            upper.append(self.observation_size[0])
                        else:
                            lower.append(int(o1))
                            upper.append(int(o1) + 1)

                        if o2 == '*':
                            lower.append(0)
                            upper.append(self.observation_size[1])
                        else:
                            lower.append(int(o2))
                            upper.append(int(o2) + 1)


                        for a1 in range(lower[0], upper[0]):
                            for a2 in range(lower[1], upper[1]):
                                for s1 in range(lower[2], upper[2]):
                                    for s2 in range(lower[3], upper[3]):
                                        for o1 in range(lower[4], upper[4]):
                                            for o2 in range(lower[5], upper[5]):
                                                self.R[s1][a1][a2][s2][o1][o2] = float(items[5].strip())


                else:
                    print('Initial letter not recognized')
                    print(line)
        self.f.close()


    def init_environment(self, batch_size):
        if not self.has_initial:
            self.current_states = np.random.choice(self.state_size, batch_size).tolist()
        else:
            self.current_states = np.random.choice(self.state_size, batch_size, p=self.initial).tolist()
        self.current_discount = 1.0
        self.current_batch_size = batch_size


    # Taken a batch of actions, returns a batch of observations and rewards in one time step
    def interact_with_environment(self, actions_0, actions_1):
        # input: actions_0[batch_size][1], actions_1[batch_size][1]
        # returns lists: observations_0[batch_size][1], observations_1[batch_size][1], rewards[batch_size][1]
        # the returned rewards have been discounted 
        observations_0 = []
        observations_1 = []
        rewards = []
        if len(actions_0) != self.current_batch_size or len(actions_1) != self.current_batch_size:
            print('batch size does not match')
        for i in range(self.current_batch_size):
            action_0 = actions_0[i]
            action_1 = actions_1[i]
            flat_list = [item for sublist in self.O[self.current_states[i]][action_0][action_1] for item in sublist]
            observation_joint = np.random.choice(self.observation_size[0] * self.observation_size[1], 1, p=flat_list)[0]
            observations_0.append(observation_joint // self.observation_size[0])
            observations_1.append(observation_joint % self.observation_size[1])

            new_state = np.random.choice(self.state_size, 1, p=self.T[self.current_states[i]][action_0][action_1])[0]
            reward = self.R[self.current_states[i]][action_0][action_1][new_state][observation_joint // self.observation_size[0]][observation_joint % self.observation_size[1]]
            self.current_states[i] = new_state
            rewards.append(reward * self.current_discount)

        # self.current_discount *= self.discount
        return observations_0, observations_1, rewards


    # DEPRECATED!
    # generate a batch of sequences at one run
    def load_samples(self, sample_size, sequence_size):
        # returns lists: actions[sample_size][sequence_size], observations[sample_size][sequence_size], rewards[sample_size][sequence_size]

        samples = []    # each output sample is a [a, a, a, ..., o, o, o, ..., r, r, r, ...] sequence
        actions = []
        observations = []
        rewards = []
        # f = open('samples.txt', 'w')
        for _ in range(sample_size):
            state = random.randint(0, self.state_size - 1)
            sample = []
            action_seq = []
            observation_seq = []
            reward_seq = []
            current_discount = 1
            for __ in range(sequence_size):
                action = random.randint(0, self.action_size - 1)
                observation = np.random.choice(self.observation_size, 1, p=self.O[state][action])[0]
                action_seq.append(action)
                observation_seq.append(observation)
                new_state = np.random.choice(self.state_size, 1, p=self.T[state][action])[0]
                reward = self.R[state][action][new_state][observation]
                reward_seq.append(reward * current_discount)       
                current_discount *= self.discount
                state = new_state
            sample = action_seq + observation_seq + reward_seq
            actions.append(action_seq)
            observations.append(observation_seq)
            rewards.append(reward_seq)
            samples.append(sample)
            
        #     for item in sample:
        #         f.write(str(item) + ' ')
        #     f.write('\n')
        # f.close()
        return actions, observations, rewards

if __name__ == "__main__":
    pass
    # dg = DecDataGenerator('grid3x3.txt')
    # # print(dg.observation_names)
    # # print(dg.action_names)
    # # # print(np.array(dg.R)[0,0,0,5,:,:])
    # dg.init_environment(2)
    # print(len(dg.initial))
    # print(dg.state_size)
    # actions_0 = [0, 1]
    # actions_1 = [2, 1]
    # observations_0, observations_1, rewards = dg.interact_with_environment(actions_0, actions_1)
    # print(observations_0)
    # print(observations_1)
    # print(rewards)