import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import gym
import math

lr = 0.0005
Capacity = 10000
num_epidose = 10000
Gamma = 0.98
lmbda = 0.95
eps_clip = 0.1

class Net(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        # self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear_actor = nn.Linear(hidden_size, output_size)
        self.Linear_critic = nn.Linear(hidden_size, 1)

    def actor_forward(self, s, dim):
        s = F.relu(self.Linear1(s))
        prob = F.softmax(self.Linear_actor(s), dim=dim)
        # print(prob)
        return prob

    def critic_forward(self, s):
        s = F.relu(self.Linear1(s))
        # s = F.relu(self.Linear2(s))
        return self.Linear_critic(s)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'rate', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):#采样
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.position = 0
        self.memory = []

class PPO(object):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPO, self).__init__()
        self.net = Net(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=Capacity)

    def act(self, s, dim):
        s = torch.Tensor(s)
        prob = self.net.actor_forward(s, dim)
        return prob

    def critic(self, s):
        return self.net.critic_forward(s)

    def put(self, s0, a0, r, s1, rate, done):
        self.buffer.push(s0, a0, r, s1, rate, done)

    def make_batch(self):
        batch = self.buffer.memory
        samples = self.buffer.memory
        batch = Transition(*zip(*samples))
        state_batch = torch.Tensor(batch.state).view(-1, 1)
        action_batch = torch.LongTensor(batch.action).view(-1, 1)
        reward_batch = torch.Tensor(batch.reward).view(-1, 1)
        next_state_batch = torch.Tensor(batch.next_state)
        rate_batch = torch.Tensor(batch.rate).view(-1, 1)
        done_batch = torch.LongTensor(batch.done).view(-1, 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, rate_batch

    def update_parameters(self):
        samples = self.buffer.memory
        batch = Transition(*zip(*samples))
        batch = self.buffer.memory
        samples = self.buffer.memory
        batch = Transition(*zip(*samples))
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1)
        reward_batch = torch.Tensor(batch.reward).view(-1, 1)
        next_state_batch = torch.Tensor(batch.next_state)
        rate_batch = torch.Tensor(batch.rate).view(-1, 1)
        done_batch = torch.LongTensor(batch.done).view(-1, 1)
        for i in range(3):
            td_target = reward_batch + Gamma * self.critic(next_state_batch) * done_batch
            delta = td_target - self.critic(state_batch)
            delta = delta.detach().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = Gamma * advantage + delta_t
                advantage_list.append(advantage)

            advantage_list.reverse()
            advantage = torch.Tensor(advantage_list)
            prob = self.act(state_batch, 1).squeeze(0)
            prob_a = prob.gather(1, action_batch.view(-1, 1))
            ratio = torch.exp(torch.log(prob_a) - torch.log(rate_batch))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(state_batch), td_target.detach())
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    Agent = PPO(env.observation_space.shape[0], 256, env.action_space.n)
    average_reward = 0
    for i_episode in range(num_epidose):
        s0 = env.reset()
        tot_reward = 0
        while True:
            env.render()
            prob = Agent.act(torch.from_numpy(s0).float(), 0)
            a0 = int(prob.multinomial(1))
            s1, r, done, _ = env.step(a0)
            rate = prob[a0].item()
            Agent.put(s0, a0, r, s1, rate, 1 - done)
            s0 = s1
            tot_reward += r
            if done:
                average_reward = average_reward + 1 / (i_episode + 1) * (
                        tot_reward - average_reward)
                if i_episode % 20 == 0:
                    print('Episode ', i_episode,
                      ' tot_reward: ', tot_reward, ' average_reward: ',
                      average_reward)
                break
        # Agent.train_net()
        Agent.update_parameters()
        Agent.buffer.clean()