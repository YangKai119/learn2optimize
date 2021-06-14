from TspSeq import *
import random
import copy

class TspModel():
    def __init__(self,problem):
        self.problem = problem

    def parse(self):
        sol = TspManager(self.problem['capacity'])
        for idx, customer in enumerate(self.problem['nodes']):
            sol.nodes.append(TspNode(node_id=idx,
                                     x=customer['position'][0],
                                     y=customer['position'][1],
                                     demand=customer['demand']))
        sol.num_nodes = len(sol.nodes)
        sol.get_dist_mat()
        return sol

    def get_random_init_solution(self):
        np.random.seed(99)
        init_sol = self.parse()
        init_rou = [i for i in range(1, init_sol.num_nodes)]
        np.random.shuffle(init_rou)
        init_rou.insert(0, 0)
        init_rou.append(0)
        print(init_rou)
        init_sol = init_sol.update_sol_state(init_sol,init_rou)
        return init_sol

    def get_greedy_init_solution(self,max_vehicle_num=1):
        sol = self.parse()
        pending_nodes = [i for i in range(1,sol.num_nodes)]
        cur_idx = 0
        cur_load = 0
        sol.add_route_node(0)
        if not max_vehicle_num:
            max_vehicle_num = sol.num_nodes

        while pending_nodes:
            nearest_next_idx = self.get_nearest_next_index(sol,pending_nodes,cur_idx,cur_load)
            # print(nearest_next_idx)
            if not nearest_next_idx:
                cur_load = 0
                if sol.route[-1] != 0:
                    sol.add_route_node(0)
                cur_idx = 0
                max_vehicle_num -= 1
            else:
                cur_load += sol.get_node(nearest_next_idx).demand
                pending_nodes.remove(nearest_next_idx)
                sol.add_route_node(nearest_next_idx)
                cur_idx = nearest_next_idx

        if not pending_nodes:
            sol.add_route_node(0)
            sol.fitness.append(sol.tot_dist)
        return sol

    def get_nearest_next_index(self,sol,pending_nodes,cur_idx,cur_load):
        nearest_idx = None
        nearest_distance = None
        for next_idx in pending_nodes:
            if cur_load + sol.get_node(next_idx).demand > sol.capacity:
                continue
            dist = sol.dist_mat[cur_idx][next_idx]
            if not nearest_distance or dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = next_idx
        return nearest_idx

    def get_random_init_solution(self):
        init_sol = self.parse()
        init_sol.route = [i for i in range(1,init_sol.num_nodes)]
        np.random.seed(0)
        np.random.shuffle(init_sol.route)
        init_sol.route.insert(0, 0)
        init_sol.route.append(0)
        init_sol.tot_dist = init_sol.get_obj()
        return init_sol