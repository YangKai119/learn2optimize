
from MkpSeq import *
import random
import copy



class MkpModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = MkpManager(self.problem['bag_capacity'],self.problem['bag_volume'])
        for idx, goods in enumerate(self.problem['goods']):
            sol.nodes.append(MkpNode(node_id=idx,
                                     weight=goods['weight'],
                                     value=goods['value'],
                                    volume=goods['volume'],
                                     num=goods['num']))
            sol.weights.extend([goods['weight']]*goods['num'])
            sol.values.extend([goods['value']]*goods['num'])
            sol.volumes.extend([goods['volume']]*goods['num'])
        sol.num_nodes = len(sol.nodes)
        return sol




