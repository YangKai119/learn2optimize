"""
根据编码的特性可知，在装载方向（0-5）上使用进化类算法效果会比较好，在装载顺序（1-N）上使用领域搜索类算法会比较好
上层用遗传算法，下层用邻域搜索算法
"""

import numpy as np
import copy
from tqdm import tqdm   # 进度条

# 种群个体类
class inds():
    def __init__(self):
        self.x = [[] for _ in range(2)]
        self.fitness = 0

    def __eq__(self, other):
        self.x = other.x
        self.fitness = other.fitness

# 用来调整装箱顺序部分的算子
class Operators():
    def node_swap(self, seq):   # 2opt
        idx1,idx2 = 0,0
        while seq[idx1] == seq[idx2]:
            idx1,idx2 = np.random.choice(range(len(seq)),2)
        seq[idx1],seq[idx2] = seq[idx2],seq[idx1]
        return seq

    def node_insert(self,seq):
        idx1 = np.random.randint(len(seq))
        val = seq[idx1]
        seq.pop(idx1)
        idx2 = np.random.randint(len(seq))
        seq.insert(idx2, val)
        return seq

    def seq_reverse(self, seq):  # 2-opt算子
        idx1, idx2 = np.random.choice(range(1, len(seq) - 1), 2, replace=False)
        if idx1 < idx2:
            new_seq = seq[:idx1] + seq[idx1:idx2][::-1] + seq[idx2:]  # 必须要保证idx1 < idx2才行
        else:
            new_seq = seq[:idx2] + seq[idx2:idx1][::-1] + seq[idx1:]
        return new_seq


class GeneticAlgorithm(Operators):     # MS部分做选择交叉变异，OS部分做邻域搜索
    def __init__(self, init_sol):
        super(GeneticAlgorithm, self).__init__()
        self.init_sol = init_sol
        self.xlen = init_sol.num_nodes
        self.N = 30  # 种群数目
        self.pop = []
        self.iterxMax = 100
        self.muta_prob = 0.2  # 变异概率
        self.cros_prob = 0.8
        self.sel_prob = 0.03

    # 初始化种群
    def init(self):
        for i in range(self.N):
            ind = inds()
            direction = [np.random.randint(0,6) for _ in range(self.init_sol.num_nodes)]
            order = [i for i in range(self.init_sol.num_nodes)]
            np.random.shuffle(order)
            ind.x = [direction,order]
            ind.fitness = self.init_sol.get_obj(ind.x)
            # print(ind.fitness)
            # print(len(self.init_sol.bins[0].goods_seq))
            self.pop.append(ind)
    # 选择过程(轮盘赌)
    def select(self):
        if np.random.rand() > self.sel_prob:
            fitness = [self.pop[i].fitness for i in range(self.N)]
            adj_fit = [1/fitness[i] for i in range(self.N)]
            # softmax计算概率归一化
            x_exp = np.exp(adj_fit)
            # 如果是列向量，则axis=0
            x_sum = np.sum(x_exp, axis=0, keepdims=True)
            fit_prob = x_exp / x_sum  # 生成种群概率
            a,b = np.random.choice(self.N, 2, replace=False, p=fit_prob)
        else:
            a,b = np.random.choice(self.N, 2, replace=False)
        return a,b

    # 交叉过程
    def crossover(self, par1, par2):                     # dirt部分使用正常交叉变异，order部分使用启发式算子做swap和reverse
        child1, child2 = inds(), inds()
        idx = np.random.randint(1,len(par1.x[0]))
        # dirt部分的交叉
        child1.x[0] = par1.x[0][:idx] + par2.x[0][idx:]
        child2.x[0] = par2.x[0][:idx] + par1.x[0][idx:]
        # order部分的交叉
        child1.x[1] = self.seq_reverse(par2.x[1])
        child2.x[1] = self.seq_reverse(par1.x[1])
        child1.fitness = self.init_sol.get_obj(child1.x)
        child2.fitness = self.init_sol.get_obj(child2.x)
        return child1, child2

    def muta(self, ind):
        times = np.random.randint(1,20)   # 最多的变异次数
        for _ in range(times):
            ind.x[0] = self.node_swap(ind.x[0])
            ind.x[1] = self.node_swap(ind.x[1])
        ind.fitness = self.init_sol.get_obj(ind.x)

    def solver(self):
        fitness = []
        self.init()
        self.pop.sort(key=lambda ind: ind.fitness,reverse=True)
        self.init_sol.init_fit = self.pop[0].fitness
        fitness.append(self.pop[0].fitness)
        # print([self.pop[i].fitness for i in range(self.N)])
        for _ in tqdm(range(self.iterxMax)):
            a, b = self.select()
            if np.random.rand() < self.cros_prob:
                child1, child2 = self.crossover(self.pop[a], self.pop[b])
                if np.random.rand() < self.muta_prob:  # 新子代a变异
                    self.muta(child1)
                if np.random.rand() < self.muta_prob:  # 新子代b变异
                    self.muta(child2)
                new = sorted([self.pop[a], self.pop[b], child1, child2], key=lambda ind: ind.fitness,reverse=True)
                self.pop[a], self.pop[b] = new[0], new[1]
            self.pop.sort(key=lambda ind: ind.fitness, reverse=True)
            fitness.append(self.pop[0].fitness)
        sol = self.init_sol.clone()
        sol.sol_seq = self.pop[0].x
        sol.best_fit = sol.get_obj(self.pop[0].x)
        # sol.best_fit = self.pop[0].fitness
        # print(sol.bins)
        sol.fitness = fitness
        return sol



