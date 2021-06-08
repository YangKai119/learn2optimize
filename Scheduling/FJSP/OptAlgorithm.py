
import numpy as np
import copy


# 种群个体类
class inds():
    def __init__(self):
        self.x = [[] for _ in range(2)]
        self.fitness = 0

    def __eq__(self, other):
        self.x = other.x
        self.fitness = other.fitness

class Operators():
    def node_swap(self, seq):
        idx1,idx2 = 0,0
        while seq[idx1] == seq[idx2]:
            idx1,idx2 = np.random.choice(range(len(seq)),2,replace=False)
        seq[idx1],seq[idx2] = seq[idx2],seq[idx1]
        return seq

    def node_insert(self,seq):
        idx1 = np.random.randint(len(seq))
        val = seq[idx1]
        seq.pop(idx1)
        idx2 = np.random.randint(len(seq))
        seq.insert(idx2, val)
        return seq

class SimulatedAnnealing(Operators):
    def __init__(self, init_sol):
        super(SimulatedAnnealing, self).__init__()
        self.T = 3e4
        self.Tmin = 1e-3
        self.alpha = 0.98
        self.init_sol = init_sol

    def solver(self):
        sol = self.init_sol.clone()
        sol.fitness.append(self.init_sol.cur_obj)
        while self.T > self.Tmin:
            # print(sol.cur_obj)
            cur_sol = sol.clone()
            cur_obj = cur_sol.cur_obj
            seq = copy.deepcopy(cur_sol.sol_seq)
            new_ms_seq = self.node_swap(seq[0])
            new_os_seq = self.node_swap(seq[1])
            new_seq = [new_ms_seq,new_os_seq]
            obj = cur_sol.get_obj(new_seq)
            delta = obj - cur_obj
            if delta > 0 and np.exp(-delta/self.T) > np.random.rand():
                cur_sol.sol_seq = new_seq
                cur_sol.cur_obj = obj
                cur_sol.fitness.append(obj)
                sol = cur_sol.clone()
            elif delta <= 0:
                cur_sol.sol_seq = new_seq
                cur_sol.cur_obj = obj
                cur_sol.fitness.append(obj)
                sol = cur_sol.clone()
            if delta <= 0:
                self.T *= self.alpha
        return sol

class GeneticAlgorithm(Operators):     # MS部分做选择交叉变异，OS部分做邻域搜索
    def __init__(self, init_sol):
        super(GeneticAlgorithm, self).__init__()
        self.init_sol = init_sol
        self.num_jobs = init_sol.num_jobs
        self.num_process = init_sol.num_process
        self.N = 30  # 种群数目
        self.pop = []
        self.iterxMax = 50000
        self.muta_prob = 0.2  # 变异概率
        self.cros_prob = 0.8
        self.sel_prob = 0.03
    # 初始化种群
    def init(self):
        for i in range(self.N):
            ind = inds()
            ind.x = self.init_sol.sol_seq
            ind.fitness = self.init_sol.cur_obj
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
    def crossover(self, par1, par2):                     # 为避免不合法的染色体出现，用各个工件的实际工序数减去后半段各工序出现的次数，得到交叉点前合法的工序数并按顺序填入
        child1, child2 = inds(), inds()
        idx = np.random.randint(1,len(par1.x[0]))
        # MS部分的交叉
        child1.x[0] = par1.x[0][:idx] + par2.x[0][idx:]
        child2.x[0] = par2.x[0][:idx] + par1.x[0][idx:]
        # OS部分的交叉
        child1.x[1] = par1.x[1][:idx] + par2.x[1][idx:]
        child2.x[1] = par2.x[1][:idx] + par1.x[1][idx:]
        stack = []  # 放入一个临时存储栈来记录不合法的染色体数，以及差多少个，或者多多少个
        for job_id in range(self.num_jobs):
            if child1.x[1].count(job_id) != self.num_process:
                stack.append(job_id)
        if len(stack) > 0:
            tmp1 = []
            tmp2 = []
            for job_id in range(self.num_jobs):
                tmp1 += [job_id] * (self.num_process - par2.x[1][idx:].count(job_id))
                tmp2 += [job_id] * (self.num_process - par1.x[1][idx:].count(job_id))
            child1.x[1] = tmp1 + par2.x[1][idx:]
            child2.x[1] = tmp2 + par1.x[1][idx:]
        child1.fitness = self.init_sol.get_obj(child1.x)
        child2.fitness = self.init_sol.get_obj(child2.x)
        return child1, child2

    # 变异过程：为了避免非法染色体的产生，变异操作采用同一染色体的两个基因交换的方式，即随机在染色体上选择两个基因位，判断这两个基因位上的基因是否相等，若相等则重新选择，若不相等则直接交换。
    def muta(self, ind):
        times = np.random.randint(1,10)   # 最多的变异次数
        for _ in range(times):
            ind.x[0] = self.node_swap(ind.x[0])
            ind.x[1] = self.node_swap(ind.x[1])
        ind.fitness = self.init_sol.get_obj(ind.x)

    def solver(self):
        sol = self.init_sol.clone()
        fitness = []
        self.init()
        for i in range(self.iterxMax):
            a, b = self.select()
            if np.random.rand() < self.cros_prob:
                child1, child2 = self.crossover(self.pop[a], self.pop[b])
                if np.random.rand() < self.muta_prob:  # 新子代a变异
                    self.muta(child1)
                if np.random.rand() < self.muta_prob:  # 新子代b变异
                    self.muta(child2)
                new = sorted([self.pop[a], self.pop[b], child1, child2], key=lambda ind: ind.fitness)
                self.pop[a], self.pop[b] = new[0], new[1]
            self.pop.sort(key=lambda ind: ind.fitness)
            fitness.append(self.pop[0].fitness)
        sol.sol_seq = self.pop[0].x
        sol.cur_obj = sol.get_obj(sol.sol_seq)
        sol.fitness = fitness
        return sol





