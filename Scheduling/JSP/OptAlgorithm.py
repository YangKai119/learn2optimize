
import numpy as np
import copy

# 种群个体类
class inds():
    def __init__(self):
        self.x = []
        self.fitness = 0

    def __eq__(self, other):
        self.x = other.x
        self.fitness = other.fitness

class Operators():
    def node_swap(self, seq):   # 2opt变异
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

    def seq_reverse(self, seq):  # 反转变异
        idx1, idx2 = np.random.choice(range(len(seq)), 2, replace=False)
        new_seq = seq[:]   # 类似深拷贝
        while new_seq == seq:
            new_seq = seq[:idx1] + seq[idx1:idx2][::-1] + seq[idx2:]
        return new_seq

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
            new_seq = self.node_swap(seq)
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
        self.sel_prob = 0.03  # 随机选择的概率
        self.one_prob = 0.5  # 模板交叉生成1的概率
    # 初始化种群
    def init(self):
        for i in range(self.N):
            ind = inds()
            ind.x = self.init_sol.sol_seq
            ind.fitness = self.init_sol.cur_obj
            self.pop.append(ind)
    # 选择过程--轮盘赌/随机选择
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
    ## 经典交叉
    def classics_crossover(self, par1, par2):                     # 为避免不合法的染色体出现，用各个工件的实际工序数减去后半段各工序出现的次数，得到交叉点前合法的工序数并按顺序填入
        child1, child2 = inds(), inds()
        idx = np.random.randint(1,len(par1.x))
        # OS部分的交叉
        child1.x = par1.x[:idx] + par2.x[idx:]
        child2.x = par2.x[:idx] + par1.x[idx:]
        stack = []  # 放入一个临时存储栈来记录不合法的染色体数，以及差多少个，或者多多少个
        for job_id in range(self.num_jobs):
            if child1.x.count(job_id) != self.num_process:
                stack.append(job_id)
        if len(stack) > 0:
            tmp1 = []
            tmp2 = []
            for job_id in range(self.num_jobs):
                tmp1 += [job_id] * (self.num_process - par2.x[idx:].count(job_id))
                tmp2 += [job_id] * (self.num_process - par1.x[idx:].count(job_id))
            child1.x = tmp1 + par2.x[idx:]
            child2.x = tmp2 + par1.x[idx:]
        child1.fitness = self.init_sol.get_obj(child1.x)
        child2.fitness = self.init_sol.get_obj(child2.x)
        return child1, child2

    ## 模板交叉
    def board_crossover(self, par1, par2):
        child1, child2 = inds(), inds()
        # 模板交叉，如果模板[i]为1时，par1[i]的基因赋值给child1，par2[i]的基因赋值给child2，反之
        board = [1 if np.random.rand() > self.one_prob else 0 for _ in range(len(par1.x))]
        for i in range(len(board)):
            if board[i] == 1:
                child1.x.append(par1.x[i])
                child2.x.append(par2.x[i])
            else:
                child1.x.append(par2.x[i])
                child2.x.append(par1.x[i])
        # 修正，避免非法工序出现，用缺失的基因替代多余的基因
        tmp1 = []
        tmp2 = []
        for job_id in range(self.num_jobs):
            if child1.x.count(job_id) < self.num_process:
                tmp1 += [job_id] * (self.num_process - child1.x.count(job_id))
            if child2.x.count(job_id) < self.num_process:
                tmp2 += [job_id] * (self.num_process - child2.x.count(job_id))
        for i in range(len(child1.x)):  # 时间消耗比较大
            job1, job2 = child1.x[i], child2.x[i]
            if child1.x.count(job1) > self.num_process:
                child1.x[i] = tmp1[0]
                tmp1.pop(0)
            if child2.x.count(job2) > self.num_process:
                child2.x[i] = tmp2[0]
                tmp2.pop(0)
        child1.fitness = self.init_sol.get_obj(child1.x)
        child2.fitness = self.init_sol.get_obj(child2.x)
        return child1, child2

    # pox交叉
    def pox_crossover(self, par1, par2):
        child1, child2 = inds(), inds()
        job_set = [i for i in range(self.num_jobs)]
        np.random.shuffle(job_set)
        idx = np.random.randint(self.num_jobs)
        # job_set1留位置，job_set2留顺序
        job_set1, job_set2 = job_set[:idx], job_set[idx:]
        # 存位置
        child1.x = [job_id if job_id not in job_set1 else -1 for job_id in par1.x]
        child2.x = [job_id if job_id not in job_set1 else -1 for job_id in par2.x]
        tmp_seq1 = par1.x[:]
        tmp_seq2 = par2.x[:]
        for job_id in job_set2:
            for _ in range(self.num_process):
                tmp_seq1.remove(job_id)
                tmp_seq2.remove(job_id)
        for idx in range(len(child1.x)):
            if child1.x[idx] == -1:
                child1.x[idx] = tmp_seq1[0]
                tmp_seq1.pop(0)
            if child2.x[idx] == -1:
                child2.x[idx] = tmp_seq2[0]
                tmp_seq2.pop(0)
        child1.fitness = self.init_sol.get_obj(child1.x)
        child2.fitness = self.init_sol.get_obj(child2.x)
        return child1, child2

    # 变异过程：为了避免非法染色体的产生，变异操作采用同一染色体的两个基因交换的方式，即随机在染色体上选择两个基因位，判断这两个基因位上的基因是否相等，若相等则重新选择，若不相等则直接交换。
    def swap_muta(self, ind):
        # times = np.random.randint(1,10)   # 最多的变异次数
        # for _ in range(times):
        ind.x = self.node_swap(ind.x)
        ind.fitness = self.init_sol.get_obj(ind.x)

    def reverse_muta(self, ind):
        # times = np.random.randint(1, 10)  # 最多的变异次数
        # for _ in range(times):
        ind.x = self.seq_reverse(ind.x)
        ind.fitness = self.init_sol.get_obj(ind.x)

    def solver(self):
        sol = self.init_sol.clone()
        fitness = []
        self.init()
        for i in range(self.iterxMax):
            a, b = self.select()
            if np.random.rand() < self.cros_prob:
                child1, child2 = self.pox_crossover(self.pop[a], self.pop[b])
                if np.random.rand() < self.muta_prob:  # 新子代a变异
                    self.swap_muta(child1)
                if np.random.rand() < self.muta_prob:  # 新子代b变异
                    self.swap_muta(child2)
                new = sorted([self.pop[a], self.pop[b], child1, child2], key=lambda ind: ind.fitness)
                self.pop[a], self.pop[b] = new[0], new[1]
            self.pop.sort(key=lambda ind: ind.fitness)
            fitness.append(self.pop[0].fitness)
        sol.sol_seq = self.pop[0].x
        sol.cur_obj = sol.get_obj(sol.sol_seq)
        sol.fitness = fitness
        return sol







