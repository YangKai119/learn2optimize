import numpy as np
import math
import pyomo.environ as pyo
from readData import *
import sys


def cal_obj(model):
    c1, c2, mu1 = 10, 1000, 1000
    tot_cost = 0
    # # 车辆运输成本
    tot_cost += c1 * sum(model.x[i, j, k] * dist_mat[i][j] for i in range(num_node) for j in range(num_node) for k in
                    range(vehicle_num) if i!=j)
    # # # 车辆启动成本
    tot_cost +=  c2 * sum(model.x[0, j, k] for j in range(1,num_node-1) for k in range(vehicle_num))
    # 车辆等待时间惩罚
    tot_cost += mu1 * sum(model.tw[i, k]  for i in range(1,num_node) for k in range(vehicle_num))


    return tot_cost


def setModel():
    # 创建模型
    model = pyo.ConcreteModel()
    # 决策变量
    model.x = pyo.Var(range(num_node), range(num_node), range(vehicle_num), domain=pyo.Binary)  # xijk
    model.r = pyo.Var(range(num_node), range(num_node), range(vehicle_num), domain=pyo.NonNegativeIntegers)  # rijk  载货量
    
    model.ta = pyo.Var(range(num_node), range(vehicle_num),bounds=(0,1500), domain=pyo.NonNegativeReals)  # 车辆k到达客户i的时间
   
    model.tw = pyo.Var(range(num_node), range(vehicle_num), domain=pyo.NonNegativeReals)  # 车辆k在客户i等待的时间
    
    model.mu = pyo.Var(range(num_node), bounds=(0, num_node), domain=pyo.NonNegativeReals)  # 消除子回路变量
    model.rho = pyo.Var(range(len(parking_pot)), bounds=(0, len(parking_pot)), domain=pyo.NonNegativeReals)  # 消除子回路变量
    model.ts = pyo.Var(range(num_node), range(len(parking_pot)), range(mobile_locker_num),
                       domain=pyo.NonNegativeReals)  # 顾客i在停放点u处开始被m服务的时间

    # 目标函数
    model.tot_cost = pyo.Objective(expr=cal_obj(model), sense=pyo.minimize)

    # 约束条件
    ## (1)每个客户点i只能被自提柜服务或车辆服务
    model.service_select = pyo.ConstraintList()
    for i in range(1,num_node-1):
        model.service_select.add(expr=sum(model.x[i,j,k] for j in range(num_node) for k in range(vehicle_num) if i != j) == 1)
        

    # ## (2)车辆与移动自提柜流平衡约束
    model.flow_balance = pyo.ConstraintList()
    for k in range(vehicle_num):
        for h in range(1, num_node-1):
            model.flow_balance.add(
                expr=sum(model.x[i, h, k] for i in range(num_node-1) if h!=i) == sum(model.x[h, j, k] for j in range(1,num_node) if h!=j))
   

    # # ## (3)消除子回路
    # model.sub_circuit = pyo.ConstraintList()
    # for i in range(num_node):
    #     for j in range(num_node):
    #         # if i == j:
    #         #     continue
    #         for k in range(vehicle_num):
    #             model.sub_circuit.add(expr=model.mu[i] - model.mu[j] + 1 - num_node * (1 - model.x[i, j, k]) <= 0)

    # 边约束
    model.arc = pyo.ConstraintList()
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                for k in range(vehicle_num):
                    model.arc.add(expr=model.x[i,j,k] == 0)

    # ## (4)车辆与移动自提柜的载重约束
    model.capacity = pyo.ConstraintList()
    for k in range(vehicle_num):
        model.capacity.add(
            expr=sum(model.x[i, j, k]*demand[i] for i in range(num_node) for j in range(num_node) if i!=j) <= vehicle_capacity)
                    
    ## (5)所有车辆或移动自提柜都从仓库出发并返回仓库
    model.start_back_depot = pyo.ConstraintList()
    for k in range(vehicle_num):
        # model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == 1)
        model.start_back_depot.add(expr=sum(model.x[j, num_node-1, k] for j in range(num_node-1)) == 1)
        # model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == sum(model.x[j, 0, k] for j in range(1, num_node)))
        model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == 1)
        model.start_back_depot.add(expr=model.x[0,num_node-1,k] == 0)
   
    ## 车辆和自提柜到达下一点的时间约束
    # 车辆从仓库出发的时间都为0
    model.arrive_time = pyo.ConstraintList()
    # for k in range(vehicle_num):
    #     model.arrive_time.add(expr=model.ta[0,k] == 0)
    
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                continue
            for k in range(vehicle_num):
                model.arrive_time.add(expr=model.ta[j, k] >=  model.ta[i, k] + model.tw[i, k] + service_time[i] + (dist_mat[i][j] / car_v) - 100000*(1-model.x[i,j,k]))
    
    # 时间窗约束
    model.arrive_due_time = pyo.ConstraintList()
    for k in range(vehicle_num):
        for i in range(1,num_node-1):
            model.arrive_due_time.add(expr=model.ta[i,k] <= time_window[i][1])
            # model.arrive_due_time.add(expr=model.ta[i,k] >= time_window[i][0])

    # # 车辆等待时间约束
    model.waiting_time_cons = pyo.ConstraintList()
    for i in range(num_node):
        for k in range(vehicle_num):
            model.waiting_time_cons.add(expr=model.tw[i, k] >= 0)
            model.waiting_time_cons.add(expr=model.tw[i, k] >= time_window[i][0] - model.ta[i, k])
    
    
    return model


# 求解器设置
def calObjFromSolvers(model, mtype, tmlimit):
    print("----Solving----")
    if mtype == 'bonmin':
        bonmin_path = '../bin/bonmin.exe'
        opt = pyo.SolverFactory('bonmin', executable=bonmin_path)  # 求解器的选择
        opt.options['bonmin.time_limit'] = tmlimit  # bonmin求解时间限制，单位秒
    elif mtype == 'glpk':
        glpk_path = '../w64/glpsol.exe'
        opt = pyo.SolverFactory('glpk', executable=glpk_path)  # 求解器的选择
        opt.options['tmlim'] = tmlimit  # glpk求解时间限制，单位秒
    elif mtype == 'grb':
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt.options['TimeLimit'] = tmlimit
    elif mtype == "msk":
        opt = pyo.SolverFactory('mosek')
    elif mtype == "ipopt":
        ipopt_path = '../bin/ipopt.exe'
        opt = pyo.SolverFactory('ipopt', executable=ipopt_path)

    results = opt.solve(model, tee=True)
    results.write()


if __name__ == "__main__":
    file_path = "../solomon/all/"
    file_name = ["c101", "c102", "c103", "c104", "c105", "c106", "c107", "c108", "c109",
                 "c201", "c202", "c203", "c204", "c205", "c206", "c207", "c208",
                 "r101", "r102", "r103", "r104", "r105", "r106", "r107", "r108", "r109", "r110", "r111", "r112",
                 "r201", "r202", "r203", "r204", "r205", "r206", "r207", "r208", "r209", "r210", "r211",
                 "rc101", "rc102", "rc103", "rc104", "rc105", "rc106", "rc107", "rc108",
                 "rc201", "rc202", "rc203", "rc204", "rc205", "rc206", "rc207", "rc208"]
    file_name = "c101.txt"
    global data_used, vehicle_num, vehicle_capacity, car_v, dist_mat, data_x_y, parking_pot, parking_dist_mat, dist_to_parking, demand, mobile_locker_num, mobile_locker_capa, mobile_locker_v, num_node, service_time, time_window
    data_used, vehicle_num, vehicle_capacity, car_v, dist_mat, data_x_y, parking_pot, parking_dist_mat, dist_to_parking, demand = get_point_data(
        file_path, file_name, 25, 5, 6,True)
    service_time = data_used['service_time'].tolist()
    num_node = len(dist_mat)
    time_window = [(data_used['tw_early'].iloc[i], data_used['tw_late'].iloc[i]) for i in range(num_node)]
    mobile_locker_num = 3  # 移动自提柜的数量
    mobile_locker_capa = 300  # 移动自提柜的容量
    mobile_locker_v = 1


    model = setModel()
    # 保存模型
    # f = open(r'..\MobileLocker\model.txt', 'w')
    # sys.stdout = f
    # model.pprint()
    # f.close()
    
    calObjFromSolvers(model, 'grb', 600)  # 6h
    print('Min Cost:', model.tot_cost())
    
    # z = []
    # for u in range(len(parking_pot)):
    #     for v in range(len(parking_pot)):
    #         for m in range(mobile_locker_num):
    #             if pyo.value(model.z[u,v,m]) == 1:
    #                 z.append((u,v,m))
                    
                    
    x = []
    for i in range(num_node):
        for j in range(num_node):
            for k in range(vehicle_num):
                if pyo.value(model.x[i,j,k]) == 1:
                    x.append((i,j,k,pyo.value(model.x[i,j,k])))
                    
    # O = []
    # for i in range(num_node):
    #     for u in range(len(parking_pot)):
    #         for m in range(mobile_locker_num):
    #             if pyo.value(model.O[i,u,m]) == 1:
    #                 O.append((i,u,m,pyo.value(model.O[i,u,m])))            
