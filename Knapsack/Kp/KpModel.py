
from KpSeq import *
import random
import copy



class KpModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = KpManager(self.problem['capacity'])
        for idx, goods in enumerate(self.problem['goods']):
            sol.nodes.append(KpNode(node_id=idx,
                                     weight = goods['weight'],
                                     value=goods['value']))
            sol.weights.append(goods['weight'])
            sol.values.append(goods['value'])
        sol.num_nodes = len(sol.nodes)
        return sol