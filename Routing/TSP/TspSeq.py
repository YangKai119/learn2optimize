import numpy as np
import copy

class TspNode(object):
    """
    Class to represent each node for vehicle routing.
    """
    def __init__(self, node_id, x, y, demand, embedding=None):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.demand = demand
        self.embedding = embedding

class SeqManager(object):
    """
    Base class for sequential input data. Can be used for vehicle routing.
    """
    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]

class TspManager(SeqManager):
    """
    The class to maintain the state for vehicle routing.
    """
    def __init__(self, capacity):
        super(TspManager, self).__init__()
        self.capacity = capacity
        self.route = []
        self.vehicle_state = []
        self.features = []
        self.tot_dist = 0
        self.dist_mat = None
        self.fitness = []

    def clone(self):
        res = TspManager(self.capacity)
        res.nodes = copy.deepcopy(self.nodes)
        res.num_nodes = copy.deepcopy(self.num_nodes)
        res.route = copy.deepcopy(self.route)
        res.features = copy.deepcopy(self.features)
        res.vehicle_state = copy.deepcopy(self.vehicle_state)
        res.dist_mat = copy.deepcopy(self.dist_mat)
        res.fitness = copy.deepcopy(self.fitness)
        return res

    def update_sol_state(self, sol, new_rou):
        new_sol = sol.clone()
        new_sol.num_nodes = 0
        new_sol.tot_dist = 0
        new_sol.route = []
        new_sol.vehicle_state = []
        new_sol.features = []
        for idx in new_rou:
            new_sol.add_route_node(idx)
        return new_sol

    def get_dist(self, node_1, node_2):
        diff_x = node_1.x - node_2.x
        diff_y = node_1.y - node_2.y
        return np.sqrt(diff_x ** 2 + diff_y ** 2)

    def get_dist_mat(self):
        self.dist_mat = np.zeros((self.num_nodes,self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    node_i = self.get_node(i)
                    node_j = self.get_node(j)
                    self.dist_mat[i][j] = self.dist_mat[j][i] = self.get_dist(node_i,node_j)

    def get_neighbor_idxes(self, route_idx):
        neighbor_idxes = []
        route_node_idx = self.vehicle_state[route_idx][0]
        pre_node_idx, pre_capacity = self.vehicle_state[route_idx - 1]
        for i in range(1, len(self.vehicle_state) - 1):
            cur_node_idx = self.vehicle_state[i][0]
            if route_node_idx == cur_node_idx:
                continue
            if pre_node_idx == 0 and cur_node_idx == 0:
                continue
            cur_node = self.get_node(cur_node_idx)
            if route_node_idx == 0 and i > route_idx and cur_node.demand > pre_capacity:
                continue
            neighbor_idxes.append(i)
        return neighbor_idxes

    def get_obj(self, seq=None):
        if not seq:
            seq = self.route
        obj = 0  # 关键路径的时间，优化目标--最大完工时间最小化
        for idx in range(len(seq)-1):
            node1 = self.get_node(seq[idx])
            node2 = self.get_node(seq[idx+1])
            obj += self.get_dist(node1, node2)
        return obj

    def add_route_node(self, node_idx):
        node = self.get_node(node_idx)
        if len(self.vehicle_state) == 0:
            pre_node_idx = 0
            pre_capacity = self.capacity
        else:
            pre_node_idx, pre_capacity = self.vehicle_state[-1]

        pre_node = self.get_node(pre_node_idx)
        cur_dist = self.get_dist(node, pre_node)
        self.tot_dist += cur_dist
        # print(node_idx, self.tot_dist)
        cur_capacity = pre_capacity - self.nodes[node_idx].demand
        if node_idx > 0:
            self.vehicle_state.append((node_idx, cur_capacity))
        else:
            self.vehicle_state.append((node_idx, self.capacity))    # depot
        depot = self.get_node(0)
        node.embedding = [node.x, node.y, node.demand * 1.0 / self.capacity, depot.x, depot.y, node.demand * 1.0 / cur_capacity, self.tot_dist]
        self.nodes[node_idx] = node    # 更新点的embedding状态
        self.features.append(node.embedding)
        self.route.append(node_idx)