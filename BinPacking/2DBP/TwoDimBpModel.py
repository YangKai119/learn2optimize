
from TwoDimBpSeq import *
import random
import copy


class BpModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = BpManager(self.problem['bin_length'],self.problem['bin_width'])
        for idx, goods in enumerate(self.problem['goods']):
            for i in range(goods['num']):
                sol.nodes.append(BpNode(node_id=len(sol.nodes),
                                         length=goods['length'],
                                         width=goods['width'],
                                         num=goods['num']))
                sol.tot_goods_areas += goods['length']*goods['width']
            sol.num_nodes += goods['num']
        return sol

    def get_random_init_solution(self):    # 随机初始解
        init_sol = self.parse()
        init_sol.direction_seq = [np.random.randint(0,2) for _ in range(init_sol.num_nodes)]
        init_sol.sol_seq.append(init_sol.direction_seq)
        init_sol.order_seq = [i for i in range(init_sol.num_nodes)]
        np.random.shuffle(init_sol.order_seq)
        init_sol.sol_seq.append(init_sol.order_seq)
        # 加入一个计算目标函数的方法
        init_sol.best_fit = init_sol.get_obj()
        init_sol.fitness.append(init_sol.best_fit)
        return init_sol
