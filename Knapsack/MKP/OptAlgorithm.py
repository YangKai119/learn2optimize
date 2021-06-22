
import numpy as np

# 种群个体类
class inds():
    def __init__(self):
        self.x = []
        self.px = []
        self.t = []
        self.pt = []
        self.fitness = 0

    def __eq__(self, other):
        self.x = other.x   # 修复后的解
        self.px = other.px   # 概率向量
        self.t = other.t   # 临时个体
        self.pt = other.pt  # 临时个体的概率向量
        self.fitness = other.fitness


class DifferentialEvolution():
    def __init__(self, init_sol):
        self.init_sol = init_sol
        self.N = 50    # 种群数目
        self.pop = []
        self.iterxMax = 400
        self.F = 0.5   # 缩放比例因子
        self.Cr = 0.5 # 交叉因子
        self.vals = init_sol.values[:]
        self.wgts = init_sol.weights[:]
        self.vols = init_sol.volumes[:]
        self.bag_capa = init_sol.bag_capacity
        self.bag_vol = init_sol.bag_volumes
        self.xlen = len(self.vals)
        self.org_idx = []

    def sort_per_val(self):               # 求单位价值大小
        per_idx = [(i, self.vals[i]/(self.wgts[i]/self.bag_capa+self.vols[i]/self.bag_vol)) for i in range(self.init_sol.num_nodes)]
        per_idx.sort(key=lambda x: x[1], reverse=True)
        # tmp_vals = self.vals[:]
        # tmp_wgts = self.wgts[:]
        for i in range(self.init_sol.num_nodes):
            idx = per_idx[i][0]
            self.org_idx.append(idx)    # 只需要得到单位价值的索引排序即可
        #     tmp_vals[i] = self.vals[idx]
        #     tmp_wgts[i] = self.wgts[idx]
        # self.vals = tmp_vals[:]
        # self.wgts = tmp_wgts[:]

    def decode(self, p):
        return [1 if p[i]>=0.5 else 0 for i in range(self.xlen)]

    def init(self):
        self.sort_per_val()    # 计算单位价值
        for _ in range(self.N):
            ind = inds()
            ind.px = [np.random.uniform() for _ in range(self.xlen)]
            ind.x = self.init_sol.repair_seq(self.decode(ind.px), self.org_idx)   # 对解进行修复，使其成为可行解
            ind.fitness = self.init_sol.get_obj(ind.x)
            self.pop.append(ind)

    def mutation(self):
        for i in range(self.N):
            cur_ind = self.pop[i]   # 当前个体
            oth_idx = [i]
            while i in oth_idx:
                oth_idx = np.random.choice(range(self.N), 3, replace=False)
            p_ind = [self.pop[idx] for idx in oth_idx]
            cur_ind.pt = [p_ind[0].px[j]+self.F*(p_ind[1].px[j]-p_ind[2].px[j]) for j in range(self.xlen)]
            cur_ind.t = self.init_sol.repair_seq(self.decode(cur_ind.px), self.org_idx)  # 变异出来的新个体

    def crossover(self):   # 连概率向量也需要交叉
        for i in range(self.N):
            cur_ind = self.pop[i]
            for j in range(self.xlen):
                if np.random.uniform() < self.Cr:
                    cur_ind.t[j] = cur_ind.x[j]
                    cur_ind.pt[j] = cur_ind.px[j]
            self.selection(cur_ind)

    def selection(self, cur_ind):    # 贪心的选择方式，x为交叉产生的新个体
        new_fitness = self.init_sol.get_obj(cur_ind.t)
        if cur_ind.fitness <= new_fitness:
            cur_ind.x = cur_ind.t[:]
            cur_ind.px = cur_ind.pt[:]
            cur_ind.fitness = new_fitness

    def solver(self):
        sol = self.init_sol.clone()
        fitness = []
        self.init()
        for i in range(self.iterxMax):
            self.mutation()
            self.crossover()
            self.pop.sort(key=lambda ind: ind.fitness, reverse=True)
            fitness.append(self.pop[0].fitness)

        sol.tot_val = self.pop[0].fitness
        sol.fitness = fitness
        sol.get_seq(self.pop[0].x)

        return sol



















