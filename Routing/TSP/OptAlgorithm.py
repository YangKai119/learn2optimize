"""
写个模拟退火和禁忌搜索
"""

import numpy as np
import copy

class Operators():   # 两种领域搜索算子，依概率选择
    def node_swap(self, seq, putIdx=False):
        idx1,idx2 = np.random.choice(range(1,len(seq)-1),2,replace=False)
        seq[idx1],seq[idx2] = seq[idx2],seq[idx1]
        if not putIdx:
            return seq
        else:
            return seq, seq[idx1], seq[idx2]
    def seq_reverse(self, seq, putIdx=False):  # 2-opt算子
        idx1, idx2 = np.random.choice(range(1,len(seq)-1), 2, replace=False)
        if idx1 < idx2:
            new_seq = seq[:idx1] + seq[idx1:idx2][::-1] + seq[idx2:]    # 必须要保证idx1 < idx2才行
        else:
            new_seq = seq[:idx2] + seq[idx2:idx1][::-1] + seq[idx1:]
        if not putIdx:
            return new_seq
        else:
            return new_seq, seq[idx1], seq[idx2]

class SimulatedAnnealing(Operators):
    def __init__(self, init_sol):
        super(SimulatedAnnealing, self).__init__()
        self.T =  3e4
        self.Tmin = 1e-3
        self.alpha = 0.98
        self.init_sol = init_sol

    def solver(self):
        sol = self.init_sol.clone()
        sol.fitness.append(self.init_sol.tot_dist)
        while self.T > self.Tmin:
            pre_obj = sol.tot_dist
            seq = copy.deepcopy(sol.route)
            if np.random.uniform() > 0.5:
                new_seq = self.node_swap(seq)
            else:
                new_seq = self.seq_reverse(seq)
            obj = sol.get_obj(new_seq)
            delta = obj - pre_obj
            if delta > 0 and np.exp(-delta/self.T) > np.random.rand():
                sol.route = new_seq
                sol.tot_dist = obj
                sol.fitness.append(obj)
            elif delta <= 0:
                sol.route = new_seq
                sol.tot_dist = obj
                sol.fitness.append(obj)
            if delta <= 0:
                self.T *= self.alpha
        return sol

class TabuSearch(Operators):
    def __init__(self, init_sol):
        super(TabuSearch, self).__init__()
        self.tabulen = 20  # 禁忌长度，禁忌表所能接受的最多禁忌对象的数量，若设置的太多则可能会造成耗时较长或者算法停止，若太少则会造成重复搜索
        self.tabu_table = [] # 禁忌表，禁忌对象为路径
        self.iterxMax = 200  # 迭代次数
        self.gbest = None
        self.neighbours = []   # 领域解、操作和适应度值一起存
        self.max_num = 1000  # 产生最大的领域解个数
        self.init_sol = init_sol
        self.spe = init_sol.tot_dist  # 特赦值，藐视准则，与obj进行对比，若obj比它小且该解已存在于禁忌表中则也可接受该解

    def update_tabu_table(self, seq, obj):
        if len(self.tabu_table) < self.tabulen:
            self.tabu_table.append([seq, obj])
        else:
            self.tabu_table.append([seq, obj])    # 禁忌表不需要排序，而是把最开始禁忌掉的方案释放
            self.tabu_table.pop(0)

    def find_neighbours(self, sol):
        self.neighbours = []
        seq = sol.route[:]
        for _ in range(self.max_num):
            nums = np.random.randint(0,10)
            new_seq = seq[:]
            for _ in range(nums):     # 设置两种领域搜索的算子来增加领域搜索能力
                if np.random.uniform() > 0.5:
                    new_seq = self.node_swap(new_seq)
                else:
                    new_seq = self.seq_reverse(new_seq)   # 都是在seq上选了两个点进行交换，此时的new_seq并不是迭代的关系
            obj = self.init_sol.get_obj(new_seq)
            if [new_seq, obj] not in self.neighbours:
                self.neighbours.append([new_seq, obj])

    def solver(self):   # 内外双层循环
        sol = self.init_sol.clone()
        sol.tot_dist = self.init_sol.tot_dist
        sol.fitness.append(self.init_sol.tot_dist)
        self.gbest = [self.init_sol.route, self.init_sol.tot_dist]
        for _ in range(self.iterxMax):
            self.find_neighbours(sol)   # 产生领域解，领域为某个解集
            self.neighbours.sort(key=lambda x: x[1])
            best_nhb_idx = 0
            best_candidate = self.neighbours[best_nhb_idx]
            found = False  # 是否找到可接受的解
            counter = 0
            while (not found) and counter < len(self.neighbours):
                if best_candidate[0] not in self.tabu_table:
                    found = True
                    # 没有在禁忌表中,接受该解,可能是次优的
                    self.tabu_table.append(best_candidate[0])
                    if best_candidate[-1] < self.spe:    # 特赦规则
                        sol.route = best_candidate[0]
                        sol.tot_dist = best_candidate[-1]
                        if best_candidate[-1] < self.gbest[-1]:   # 更新全局最优解
                            self.gbest = best_candidate[:]
                else:
                    # 藐视准则,该领域解是目前最好的解
                    if best_candidate[-1] <= self.gbest[-1]:
                        self.gbest = best_candidate[:]
                        sol.route = self.gbest[0]
                        sol.tot_dist = self.gbest[1]
                        found = True
                        # print("接受特赦解")
                    # 没有最优,选择次优
                    else:
                        best_nhb_idx += 1
                        best_candidate = self.neighbours[best_nhb_idx]
                counter += 1

            if len(self.tabu_table) >= self.tabulen:
                self.tabu_table.pop(0)
                # print("禁忌表满")
            sol.fitness.append(sol.tot_dist)
        return sol







