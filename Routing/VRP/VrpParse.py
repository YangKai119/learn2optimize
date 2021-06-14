
import numpy as np

class VrpNode(object):
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

class VrpManager(SeqManager):
    """
    The class to maintain the state for vehicle routing.
    """
    def __init__(self, capacity, demands):
        super(VrpManager, self).__init__()
        self.capacity = capacity
        self.demands = demands
        self.route = []
        self.tot_dist = 0
        self.dist_mat = None
        self.fitness = []
        self.num_vehi = len(self.capacity)

    def get_dist(self, node_1, node_2):
        diff_x = node_1.x - node_2.x
        diff_y = node_1.y - node_2.y
        return np.sqrt(diff_x ** 2 + diff_y ** 2)

    def get_dist_mat(self):
        self.dist_mat = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if i != j:
                    node_i = self.get_node(i)
                    node_j = self.get_node(j)
                    self.dist_mat[i][j] = self.dist_mat[j][i] = self.get_dist(node_i, node_j)

class Parse():
    def __init__(self, problem):
        self.problem = problem
        self.capacity = [250]*3 + [150]*5 + [300]*2  # 共10辆车
        self.demands = problem['demand'].tolist()

    def parse(self):
        sol = VrpManager(self.capacity, self.demands)
        for idx in range(len(self.problem)):
            customer = self.problem.iloc[idx]
            sol.nodes.append(VrpNode(node_id=idx,
                                     x=customer['x_coord'],
                                     y=customer['y_coord'],
                                     demand=customer['demand']))
        sol.num_nodes = len(sol.nodes)
        sol.get_dist_mat()
        return sol

