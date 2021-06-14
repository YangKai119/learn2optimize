import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple

GAMMA = 0.99
lr = 0.1
EPSION = 0.1
buffer_size = 10000  # replay池的大小
batch_size = 32
num_episode = 100000
target_update = 10  # 每过多少个episode将net的参数复制到target_net


# 定义神经网络
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print('x: ', x)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x


# nametuple容器
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):  # 采样
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size)
        self.target_net = Net(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = ReplayMemory(buffer_size)
        self.loss_func = nn.MSELoss()
        self.steps_done = 0

    def put(self, s0, a0, r, t, s1):
        self.buffer.push(s0, a0, r, t, s1)

    def select_action(self, state):
        eps_threshold = random.random()
        action = self.net(torch.Tensor(state))
        if eps_threshold > EPSION:
            choice = torch.argmax(action).numpy()
        else:
            choice = np.random.randint(0, action.shape[
                0])  # 随机[0, action.shape[0]]之间的数
        return choice

    def update_parameters(self):
        if self.buffer.__len__() < batch_size:
            return
        samples = self.buffer.sample(batch_size)
        batch = Transition(*zip(*samples))
        # 将tuple转化为numpy
        tmp = np.vstack(batch.action)
        # 转化成Tensor
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.LongTensor(tmp.astype(int))
        reward_batch = torch.Tensor(batch.reward)
        done_batch = torch.Tensor(batch.done)
        next_state_batch = torch.Tensor(batch.next_state)

        q_next = torch.max(self.target_net(next_state_batch).detach(), dim=1,
                           keepdim=True)[0]
        q_eval = self.net(state_batch).gather(1, action_batch)
        q_tar = reward_batch.unsqueeze(1) + (1-done_batch) * GAMMA * q_next
        loss = self.loss_func(q_eval, q_tar)
        # print(loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # 状态空间：4维
    # 动作空间：1维，并且是离散的，只有0和1两个动作
    Agent = DQN(env.observation_space.shape[0], 256, env.action_space.n)
    average_reward = 0  # 目前所有的episode的reward的平均值
    for i_episode in range(num_episode):
        s0 = env.reset()
        tot_reward = 0  # 每个episode的总reward
        tot_time = 0  # 实际每轮运行的时间 （reward的定义可能不一样）
        while True:
            env.render()
            a0 = Agent.select_action(s0)
            s1, r, done, _ = env.step(a0)
            tot_time += r  # 计算当前episode的总时间
            # 网上定义的reward方法
            # x, x_dot, theta, theta_dot = s1
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            tot_reward += r  # 计算当前episode的总reward
            if done:
                t = 1
            else:
                t = 0
            Agent.put(s0, a0, r, t, s1)  # 放入replay池
            s0 = s1
            Agent.update_parameters()
            if done:
                average_reward = average_reward + 1 / (i_episode + 1) * (
                        tot_reward - average_reward)
                print('Episode ', i_episode, 'tot_time: ', tot_time,
                      ' tot_reward: ', tot_reward, ' average_reward: ',
                      average_reward)
                break
        if i_episode % target_update == 0:
            Agent.target_net.load_state_dict(Agent.net.state_dict())

