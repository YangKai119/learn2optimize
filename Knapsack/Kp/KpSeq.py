

import numpy as np
import copy

class KpNode(object):
    def __init__(self, node_id, weight, value, embedding=None):
        self.node_id = node_id
        self.weight = weight
        self.value = value
        self.embedding = embedding

class SeqManager(object):

    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]

class KpManager(SeqManager):

    def __init__(self, capacity):
        super(KpManager, self).__init__()
        self.capacity = capacity
        self.goods_seq = []
        self.knapsack_state = []
        self.weights = []
        self.values = []
        self.features = []
        self.tot_obj = 0
        self.fitness = []

    def clone(self):
        res = KpManager(self.capacity)
        res.nodes = copy.deepcopy(self.nodes)
        res.num_nodes = copy.deepcopy(self.num_nodes)
        res.features = copy.deepcopy(self.features)
        res.knapsack_state = copy.deepcopy(self.knapsack_state)
        res.fitness = copy.deepcopy(self.fitness)
        res.weights = copy.deepcopy(self.weights)
        res.values = copy.deepcopy(self.values)
        return res

    def get_seq(self, visit):
        self.goods_seq = []
        for i in range(self.num_nodes):
            if visit[i] != 0:
                self.goods_seq.append(i)

    def get_obj(self, seq=None):
        if not seq:
            seq = self.goods_seq
        obj = 0  # 关键路径的时间，优化目标--最大完工时间最小化
        for idx in seq:
            node = self.get_node(idx)
            obj += node.value
        return obj

