
"""
根据编码的特性可知，在装载方向（0，1，2）上使用进化类算法效果会比较好，在装载顺序（1-N）上使用领域搜索类算法会比较好
上层用人工免疫算法，下层用领域搜索算法
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

# 人工免疫算法
class ImmuneAlgorithm(Operators):
    def __init__(self, init_sol):
        super(ImmuneAlgorithm, self).__init__()
        self.iterxMax = 100
        self.N = 30    # 种群数
        self.pop = []
        self.ncl = 5  # 克隆数
        self.init_sol = init_sol
        self.xlen = init_sol.num_nodes

    # 初始化种群
    def init(self):
        for i in range(self.N):
            ind = inds()
            direction = [np.random.randint(0, 2) for _ in range(self.xlen)]
            order = [i for i in range(self.xlen)]
            np.random.shuffle(order)
            ind.x = [direction, order]
            ind.fitness = self.init_sol.get_obj(ind.x)
            self.pop.append(ind)

    def muta(self, ind):
        if np.random.uniform() < 0.5:
            ind.x[0] = self.node_swap(ind.x[0])
            ind.x[1] = self.node_swap(ind.x[1])
        else:
            ind.x[0] = self.seq_reverse(ind.x[0])
            ind.x[1] = self.seq_reverse(ind.x[1])
        ind.fitness = self.init_sol.get_obj(ind.x)
        return ind

    # 免疫操作函数：克隆、变异、变异抑制
    def variation(self):
        var_inds = []  # 存储变异后的个体
        for i in range(np.int(self.N / 2)):  # 遍历前一半个体
            # 选激励度前NP/2个体进行免疫操作
            ind = self.pop[i]  # 当前个体
            clone_ind = [ind]*(self.ncl+1)  # 对当前个体进行克隆 Na.shape=(Ncl, city_num)
            # 保留克隆源个体
            # 克隆抑制，保留亲和度最高的个体
            muta_ind = []  # 存储变异种群亲和度值
            for j in range(1,self.ncl):  # 从1开始，0为原体，遍历每一个克隆样本
                muta_ind.append(self.muta(clone_ind[j]))  # 亲和度为目标值
            muta_ind.sort(key=lambda ind: ind.fitness, reverse=True)  # 目标为求最大空间利用率，所以激励度按升序排序
            var_inds.append(muta_ind[0])     # 取最优的传入下一代
        return var_inds

    # 创建新生种群
    def refresh(self):
        new_pop = []
        for i in range(np.int(self.N / 2)):  # 遍历每一个个体
            ind = inds()
            direction = [np.random.randint(0, 2) for _ in range(self.xlen)]
            order = [i for i in range(self.xlen)]
            np.random.shuffle(order)
            ind.x = [direction, order]
            ind.fitness = self.init_sol.get_obj(ind.x)
            new_pop.append(ind)
        return new_pop

    def solver(self):
        self.init()
        self.pop.sort(key=lambda ind: ind.fitness, reverse=True)
        fitness = [self.pop[0].fitness]
        self.init_sol.init_fit = self.pop[0].fitness
        for _ in tqdm(range(self.iterxMax)):
            var_inds = self.variation()    # 效率太低，有优化的空间
            new_pop = self.refresh()
            self.pop = var_inds + new_pop
            self.pop.sort(key=lambda ind: ind.fitness, reverse=True)
            fitness.append(self.pop[0].fitness)
        sol = self.init_sol.clone()
        sol.sol_seq = self.pop[0].x
        sol.best_fit = sol.get_obj(self.pop[0].x)
        # sol.best_fit = self.pop[0].fitness
        # print(sol.bins)
        sol.fitness = fitness
        return sol












