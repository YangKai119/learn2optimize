import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
from torch.distributions import Categorical
from collections import deque
from collections import namedtuple

GAMMA = 1.0
lr = 0.1
EPSION = 0.9
buffer_size = 10000
batch_size = 32
num_episode = 100000
target_update = 10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        # self.Linear2 = nn.Linear(hidden_size, hidden_size)
        # self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        # x = F.relu(self.Linear2(x))
        x = F.softmax(self.Linear3(x), dim=1)
        return x
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return F.softmax(x, dim=1)

class Reinforce(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Policy(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.net.parameters(), lr=0.01)

    def select_action(self, s):
        s = torch.Tensor(s).unsqueeze(0)
        probs = self.net(s)
        tmp = Categorical(probs)
        a = tmp.sample()
        log_prob = tmp.log_prob(a)
        return a.item(), log_prob

    def update_parameters(self, rewards, log_probs):
        R = 0
        loss = 0
        # for i in reversed(range(len(rewards))):
        #     R = rewards[i] + GAMMA * R
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            loss = loss - R * log_probs[i]
        # discounts = [GAMMA ** i for i in range(len(rewards) + 1)]
        # R = sum([a * b for a, b in zip(discounts, rewards)])
        # policy_loss = []
        # for log_prob in log_probs:
        #     policy_loss.append(-log_prob * R)
        # loss = torch.cat(policy_loss).sum()
        # print('loss: ', len(loss))
        # loss = loss / len(loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    average_reward = 0
    Agent = Reinforce(env.observation_space.shape[0], 16, env.action_space.n)
    # scores_deque = deque(maxlen=100)
    # scores = []
    for i_episode in range(1, num_episode + 1):
        s = env.reset()
        log_probs = []
        rewards = []
        while True:
            env.render()
            a, prob = Agent.select_action(s)
            s1, r, done, _ = env.step(a)
            # scores_deque.append(sum(rewards))
            # scores.append(sum(rewards))
            log_probs.append(prob)
            rewards.append(r)
            s = s1
            if done:
                average_reward = average_reward + (1 / (i_episode + 1)) * (np.sum(rewards) - average_reward)
                if i_episode % 100 == 0:
                    # print('Episode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                    print('episode: ', i_episode, "tot_rewards: ", np.sum(rewards), 'average_rewards: ', average_reward)
                break
        Agent.update_parameters(rewards, log_probs)