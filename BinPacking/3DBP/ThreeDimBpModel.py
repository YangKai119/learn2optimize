
from ThreeDimBpSeq import *
import random
import copy


class BpModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = BpManager(self.problem['bin_capacity'],self.problem['bin_length'],self.problem['bin_width'],self.problem['bin_height'])
        for idx, goods in enumerate(self.problem['goods']):
            for i in range(goods['num']):
                sol.nodes.append(BpNode(node_id=len(sol.nodes),
                                         weight=goods['weight'],
                                         length=goods['length'],
                                         width=goods['width'],
                                         height = goods['height'],
                                         num=goods['num']))
                sol.tot_goods_vols += goods['height']*goods['length']*goods['width']
            sol.weights.extend([goods['weight']]*goods['num'])
        sol.num_nodes = len(sol.weights)
        return sol

    def get_random_init_solution(self):    # 随机初始解
        init_sol = self.parse()
        init_sol.direction_seq = [np.random.randint(0,6) for _ in range(init_sol.num_nodes)]
        init_sol.sol_seq.append(init_sol.direction_seq)
        init_sol.order_seq = [i for i in range(init_sol.num_nodes)]
        np.random.shuffle(init_sol.order_seq)
        init_sol.sol_seq.append(init_sol.order_seq)
        # 加入一个计算目标函数的方法
        init_sol.best_fit = init_sol.get_obj()
        init_sol.fitness.append(init_sol.best_fit)
        return init_sol
