# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:31:39 2021

@author: omgya
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

class OR_Tools():
    
    def __init__(self,cust_df,vehicles_df,dist_mat,drv_time_mat):
        #参数初始化
        self.vehicles_df = vehicles_df
        self.vehicles = dict(zip(vehicles_df.index,vehicles_df[['DIST_STATION_CODE','VEHICLE_CODE','MAX_LOAD_PACKAGE','MAX_LOAD_CUST']].values))
        self.cust_df = cust_df
        self.dist_mat = dist_mat
        self.drv_time_mat = drv_time_mat
    
    def create_data_model(self):
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = self.dist_mat
        data['drv_time_matrix'] = self.drv_time_mat
        #需求点包含了发货点数据
        data['demands'] = self.cust_df['QTY_ORDER_AVG'].tolist()
        data['vehicle_capacities'] = [int(x*0.95) for x in self.vehicles_df['MAX_LOAD_PACKAGE']] #每辆车不能完全装满
        data['num_vehicles'] = len(data['vehicle_capacities'])
        #data['depot'] = 0
        #data['starts'] = list(np.zeros(data['num_vehicles']).astype(np.int64))
        data['starts'] = [0] * data['num_vehicles']
        data['ends'] = [0] * data['num_vehicles']
        #data['ends'] = list(np.zeros(data['num_vehicles']).astype(np.int64))
        return data
    
    
    def print_solution(self,data,manager,routing,solution):
        """Prints solution on console."""
        # Display dropped nodes.
        dropped_nodes = 'Dropped nodes:'
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if solution.Value(routing.NextVar(node)) == node:
                dropped_nodes += ' {}'.format(manager.IndexToNode(node))
        print(dropped_nodes)
        # Display routes
        total_distance = 0
        total_load = 0
        total_time = []
        route_plan = []
        for vehicle_id in self.vehicles_df.index:
            vehicle_code = self.vehicles_df['VEHICLE_CODE'].loc[vehicle_id]
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_code)
            route_distance = 0
            route_load = 0
            route_nodes = []      #输出每辆车的路线
            route_time = 0
            sum_load_r = 0
            driving_time = 0
            service_time = 0
            while not routing.IsEnd(index):     #当routing.IsEnd(index)为True时，index那个点没去到
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, round(route_load,2))
                route_nodes.append(node_index)
                previous_index = index          #当前节点
                index = solution.Value(routing.NextVar(index))          #下一到达节点
                route_distance += routing.GetArcCostForVehicle(previous_index, 
                                                               index, 
                                                               vehicle_id)
            #将起点加入路线    
            route_nodes.append(0)
            #计算路线时间
            for i in range(len(route_nodes)-1):
                driving_time += self.drv_time_mat[route_nodes[i]][route_nodes[i+1]]
                service_time += self.cust_df['SERVICE_TIME'].iloc[route_nodes[i]]
                    
            route_time = driving_time + service_time*60
            
            plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                     round(route_load,2))
            plan_output += 'Distance of the route: {}km\n'.format(route_distance/1000)
            plan_output += 'Load of the route: {}\n'.format(round(route_load,2))
            load_r = route_load/self.vehicles[vehicle_id][2]
            plan_output += 'Loading rate of the route: {}\n'.format(round(load_r,2))
            plan_output += 'Time of the route: {}h\n'.format(round(route_time/3600,2))
            self.vehicles_df['DRIVE_TIME'].loc[vehicle_id] += round(route_time/3600,2)
            plan_output += 'Customers of the route: {}\n'.format(len(route_nodes)-2)
            print(plan_output)
            
            total_distance += route_distance
            total_time.append(route_time)
            total_load += route_load
            #sum_load_r += load_r
            #avg_load_r = sum_load_r/data['num_vehicles']
            route_plan.append(route_nodes)
        print('Total distance of all routes: {}km'.format(total_distance/1000))
        print('Total load of all routes: {}'.format(round(total_load,2)))
        print('Total time of all routes: {}h'.format(round(max(total_time)/3600,2)))     #总路线（各区域）时间，应该是取所有车的路线最大时间作为该次规划的路线总时间
        #print('Average loading rate of all routes: {}'.format(round(avg_load_r,2)))
        return route_plan
    
    
    def main(self):
        """Solve the CVRP problem."""
        # Instantiate the data problem.
        data = self.create_data_model()
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'],
                                               data['starts'],data['ends'])
        
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
    
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        #总驾驶时间回调
        def drv_time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            service_time = self.cust_df['SERVICE_TIME'].iloc[to_node] if to_node!=0 else 0
            drv_time = data['drv_time_matrix'][from_node][to_node]
            return drv_time + service_time
        
        #RegisterTransitCallback建立约束条件
        drv_time_callback_index = routing.RegisterTransitCallback(drv_time_callback)
        deliver_time = 3600*8    #一天工作8小时
        #定义每辆车的最大总驾驶时间
        dimension_name = 'Driving_time'
        routing.AddDimension(
            drv_time_callback_index,
            int(3600*0.5),  # no slack    松弛在半个小时以内
            (deliver_time+int(3600*0.5)),  # vehicle maximum travel time
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(int(10))

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    
        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
    
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacity = "Capacity"
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
        
        # 最大满载率约束
        capacity_dimension = routing.GetDimensionOrDie(capacity)
        mini_load_r = 0.8  #最低满载率
        mini_load = list(np.array(data['vehicle_capacities'])*mini_load_r)
        for vehicle in range(data['num_vehicles']):             #限制的是每辆车
            capacity_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle),
                                                         int(mini_load[vehicle]),   #车辆装载量
                                                         int(mini_load_r))
    
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)              #自动获取初始解
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)        #用禁忌搜索算法进行求解
        search_parameters.time_limit.FromSeconds(120)         #单次运行时间限制
    
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
    
        # Print solution on console.
        if solution:
            route_plan = self.print_solution(data,manager,routing,solution)
        
        return route_plan,self.vehicles_df