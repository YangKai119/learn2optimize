
import numpy as np
import copy
# 种群个体类
class inds():
    def __init__(self):
        self.x = []
        self.v = []  # 初始化速度

    def __eq__(self, other):
        self.x = other.x
        self.v = other.v

class ParticleSwarmOptimization():
    def __init__(self, init_sol):
        self.w = 0.8
        self.c1 = 1.5
        self.c2 = 1.5
        self.r1 = 0.7
        self.r2 = 0.3
        self.iterxMax = 5000
        self.N = 30 # 种群规模
        self.g_best = init_sol.sol_seq  # 初始化全局最优解
        self.g_fitness = init_sol.tot_benfit
        self.num_worker = init_sol.num_worker
        self.num_tasks = init_sol.num_tasks
        self.init_sol = init_sol
        self.pop = []
        self.p_best = []
        self.p_fitness = []

    def init(self):
        for i in range(self.N):
            ind = inds()
            ind.x = self.init_sol.sol_seq
            ind.v = [np.random.uniform() for _ in range(self.num_tasks)]
            fitness = self.init_sol.tot_benfit
            self.pop.append(ind)
            self.p_best.append(ind.x)
            self.p_fitness.append(fitness)

    def encode(self, x_new):
        same_num = self.num_tasks - self.num_worker  # 指派同一个工人工作的任务数量，有几个框框
        x_new_u = [(i, x_new[i]) for i in range(self.num_tasks)]   # 保存索引，后期逆序映射时使用
        x_sort = sorted(x_new_u, key=lambda x: x[1])  # 按值排序，且不改变原始list
        x_adj = [(i, x_sort[i+1][1]-x_sort[i][1], x_sort[i][1], x_sort[i+1][1]) for i in range(self.num_tasks-1)]  # 邻位相减，记录减数与被减数
        x_adj_idx = sorted(x_adj, key=lambda x: x[1])[:same_num]  # 升序排列
        x_adj_idx.sort(key=lambda x: x[0])
        x_tmp_idx = [[x_adj_idx[0][2],x_adj_idx[0][3]]]
        # 生成抱团块，做一个区间合并，在相同区间的指派同一个工人，若后期需要限制工人的最大任务数量可以从此处修改
        for i in range(1,same_num):
            if x_tmp_idx[-1][-1] == x_adj_idx[i][2]:
                x_tmp_idx[-1].append(x_adj_idx[i][3])
            else:
                x_tmp_idx.append([x_adj_idx[i][2],x_adj_idx[i][3]])
        worker_ids = []
        worker_id = 0
        i = 0   # 位置的索引
        idx = 0  # x_tmp_idx的索引，区间内的worker_id = worker_id + idx
        while len(worker_ids) < self.num_tasks:
            if x_sort[i][1] in x_tmp_idx[idx]: # 遇上组团的，直接更新
                worker_ids.extend([worker_id]*len(x_tmp_idx[idx]))
                worker_id += 1
                i += len(x_tmp_idx[idx])
                idx = idx + 1 if idx != len(x_tmp_idx)-1 else idx  # 防止数组越界
            else:
                worker_ids.append(worker_id)
                worker_id += 1
                i += 1
        # 逆序映射
        new_x = [0] * self.num_tasks
        for i in range(self.num_tasks):
            item = x_sort[i]
            org_idx = item[0]
            new_x[org_idx] = worker_ids[i]
        return new_x

    def solver(self):
        sol = self.init_sol
        fitness = []
        self.init()
        # 迭代次数
        for _ in range(self.iterxMax):
            for i in range(self.N):
                # print(self.pop[i].x)
                tmp = self.init_sol.get_obj(self.pop[i].x)
                if tmp > self.p_fitness[i]:
                    self.p_best[i] = self.pop[i].x
                    self.p_fitness[i] = tmp
                    if self.p_fitness[i] > self.g_fitness:
                        self.g_fitness = self.p_fitness[i]
                        self.g_best = self.p_best[i]
            # 速度和位置更新
            for i in range(self.N):
                self.pop[i].v = [self.w * self.pop[i].v[idx] + self.c1 * self.r1 * (self.p_best[i][idx] - self.pop[i].x[idx]) + self.c2 * self.r2 * (self.g_best[idx] - self.pop[i].x[idx]) for idx in range(self.num_tasks)]
                self.pop[i].x = [self.pop[i].v[idx] + self.pop[i].x[idx] for idx in range(self.num_tasks)]
                self.pop[i].x = self.encode(self.pop[i].x)
            sol.sol_seq = self.g_best[:]
            sol.tot_benfit = self.g_fitness
            fitness.append(self.g_fitness)
        sol.fitness = fitness[:]
        return sol











