import numpy as np
import copy

class MkpNode(object):
    def __init__(self, node_id, weight, volume, value, num, embedding=None):
        self.node_id = node_id
        self.weight = weight
        self.value = value
        self.volume = volume
        self.num = num
        self.embedding = embedding

class SeqManager(object):

    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]

class MkpManager(SeqManager):

    def __init__(self, bag_capacity, bag_volumes):
        super(MkpManager, self).__init__()
        self.bag_capacity = bag_capacity
        self.bag_volumes = bag_volumes
        self.goods_seq = []
        self.knapsack_state = []
        self.weights = []
        self.volumes = []
        self.values = []
        self.features = []
        self.tot_val = 0
        self.fitness = []

    def clone(self):
        res = MkpManager(self.bag_capacity, self.bag_volumes)
        res.nodes = copy.deepcopy(self.nodes)
        res.num_nodes = copy.deepcopy(self.num_nodes)
        res.features = copy.deepcopy(self.features)
        res.knapsack_state = copy.deepcopy(self.knapsack_state)
        res.fitness = copy.deepcopy(self.fitness)
        res.weights = copy.deepcopy(self.weights)
        res.values = copy.deepcopy(self.values)
        return res

    def get_seq(self, visit):         # visit --> [1,0,1,0,...,0,1] 二进制的形式
        self.goods_seq = []
        for i in range(self.num_nodes):
            if visit[i] != 0:
                self.goods_seq.append(i)

    def get_obj(self, visit=None):
        if visit:
            self.get_seq(visit)
        seq = self.goods_seq
        obj = 0  # 关键路径的时间，优化目标--最大完工时间最小化
        for idx in seq:
            node = self.get_node(idx)
            obj += node.value
        return obj

    # 优化违约解，按pre_vals从大到小的顺序来决定把多余的1去掉，若本身为0则不用理会
    def repair_seq(self, visit, org_idx):     # org_idx是按单位价值排序后返回的原始索引排序
        tot_wgt = 0
        tot_vol = 0
        for i in range(self.num_nodes):
            idx = org_idx[i]
            tmp_wgt = tot_wgt + self.weights[idx]   # 用来判断约束条件
            tmp_vol = tot_vol + self.volumes[idx]
            if visit[idx] == 1 and tmp_wgt < self.bag_capacity and tmp_vol < self.bag_volumes:
                tot_wgt += self.weights[idx]
                tot_vol += self.volumes[idx]
            else:
                visit[idx] = 0

        return visit





