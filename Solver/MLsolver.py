import numpy as np
import math
import pyomo.environ as pyo
from readData import *
import sys
import time

def cal_obj(model,alpha,beta):
    c1, c2, c3, c4, mu1, mu2 = 10, 1000, 15, 2000, alpha, beta
    tot_cost = 0
    # 车辆运输成本
    tot_cost += c1 * sum(model.x[i, j, k] * dist_mat[i][j] for i in range(num_node) for j in range(num_node) for k in
                    range(vehicle_num) if i!=j)
    # 车辆启动成本
    tot_cost +=  c2 * sum(model.x[0, j, k] for j in range(1,num_node) for k in range(vehicle_num))
    # # 自提柜运输成本
    tot_cost += c3 * sum(
        model.z[u, v, m] * parking_dist_mat[u][v] for u in range(1,len(parking_pot)) for v in range(len(parking_pot))
        for m in range(mobile_locker_num))
    # 自提柜的启动成本
    tot_cost += c4 * sum(model.z[0, v, m] for v in range(1,len(parking_pot)) for m in range(mobile_locker_num))
    # 车辆等待时间惩罚
    tot_cost += mu1 * sum(model.tw[i, k]  for i in range(1,num_node) for k in range(vehicle_num))
    # 移动自提柜的闲置时间惩罚
    tot_cost += mu2 * sum(model.ttw[u, m]  for u in range(1,len(parking_pot)) for m in range(mobile_locker_num))

    return tot_cost


def setModel(L, alpha, beta):
    # 创建模型
    model = pyo.ConcreteModel()
    # 决策变量
    model.x = pyo.Var(range(num_node), range(num_node), range(vehicle_num), domain=pyo.Binary)  # xijk
    model.z = pyo.Var(range(len(parking_pot)), range(len(parking_pot)), range(mobile_locker_num),
                      domain=pyo.Binary)  # zuvm
    # model.y = pyo.Var(range(num_node), range(vehicle_num), domain=pyo.Binary)  # yki
    model.O = pyo.Var(range(num_node), range(len(parking_pot)), range(mobile_locker_num), domain=pyo.Binary)  # Oium
    # model.r = pyo.Var(range(num_node), range(num_node), range(vehicle_num), domain=pyo.NonNegativeIntegers)  # rijk  载货量
    # model.g = pyo.Var(range(len(parking_pot)), range(len(parking_pot)), range(mobile_locker_num),
    #                   domain=pyo.NonNegativeIntegers)  # guvm
    model.ta = pyo.Var(range(num_node), range(vehicle_num), domain=pyo.NonNegativeReals)  # 车辆k到达客户i的时间
    model.tta = pyo.Var(range(len(parking_pot)), range(mobile_locker_num),domain=pyo.NonNegativeReals)  # 自提柜m到达停放点u的时间
    model.tw = pyo.Var(range(num_node), range(vehicle_num), domain=pyo.NonNegativeReals)  # 车辆k在客户i等待的时间
    model.ttw = pyo.Var(range(len(parking_pot)), range(mobile_locker_num),
                        domain=pyo.NonNegativeReals)  # 自提柜m在停放点u闲置的时间
    model.ttl = pyo.Var(range(len(parking_pot)), range(mobile_locker_num), domain=pyo.NonNegativeReals)  # 自提柜m离开停放点u的时间
    model.mu = pyo.Var(range(num_node), bounds=(0, num_node), domain=pyo.NonNegativeReals)  # 消除子回路变量
    model.rho = pyo.Var(range(len(parking_pot)), bounds=(0, len(parking_pot)), domain=pyo.NonNegativeReals)  # 消除子回路变量
    model.ts = pyo.Var(range(num_node), range(len(parking_pot)), range(mobile_locker_num),
                       domain=pyo.NonNegativeReals)  # 顾客i在停放点u处开始被m服务的时间

    # 目标函数
    model.tot_cost = pyo.Objective(expr=cal_obj(model, alpha, beta), sense=pyo.minimize)

    # 约束条件
    ## (1)每个客户点i只能被自提柜服务或车辆服务
    model.service_select = pyo.ConstraintList()
    for i in range(1,num_node-1):
        model.service_select.add(expr=sum(model.x[i,j,k] for j in range(num_node) for k in range(vehicle_num)) + sum(
            model.O[i, u, m] for u in range(len(parking_pot)) for m in range(mobile_locker_num)) == 1)
        # 只有车辆服务
        # model.service_select.add(expr=sum(model.x[i,j,k] for j in range(num_node) for k in range(vehicle_num)) == 1)
        

    # ## (2)车辆与移动自提柜流平衡约束
    model.flow_balance = pyo.ConstraintList()
    for k in range(vehicle_num):
        for h in range(1, num_node-1):
            model.flow_balance.add(
                expr=sum(model.x[i, h, k] for i in range(num_node-1) if h!=i) == sum(model.x[h, j, k] for j in range(1,num_node) if h!=j))
    
    for m in range(mobile_locker_num):
        for g in range(1, len(parking_pot)-1):
            model.flow_balance.add(
                expr=sum(model.z[u, g, m] for u in range(len(parking_pot)-1) if g!=u) == sum(
                    model.z[g, v, m] for v in range(1,len(parking_pot)) if g!=v))
    
    # 边约束
    model.arc = pyo.ConstraintList()
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                for k in range(vehicle_num):
                    model.arc.add(expr=model.x[i,j,k] == 0)
    
    for u in range(len(parking_pot)):
        for v in range(len(parking_pot)):
            if u == v:
                for m in range(mobile_locker_num):
                    model.arc.add(expr=model.z[u,v,m] == 0)
    

    # ## (3)消除子回路
    # model.sub_circuit = pyo.ConstraintList()
    # for i in range(num_node):
    #     for j in range(num_node):
    #         # if i == j:
    #         #     continue
    #         for k in range(vehicle_num):
    #             model.sub_circuit.add(expr=model.mu[i] - model.mu[j] + 1 - num_node * (1 - model.x[i, j, k]) <= 0)

    # for u in range(len(parking_pot)):
    #     for v in range(len(parking_pot)):
    #         # if u == v:
    #         #     continue
    #         for m in range(mobile_locker_num):
    #             model.sub_circuit.add(
    #                 expr=model.rho[u] - model.rho[v] + 1 - len(parking_pot) * (1 - model.z[u, v, m]) <= 0)

    # ## (4)车辆与移动自提柜的载重约束
    model.capacity = pyo.ConstraintList()
    for k in range(vehicle_num):
        model.capacity.add(
            expr=sum(model.x[i, j, k]*demand[i] for i in range(num_node) for j in range(num_node) if i!=j) <= vehicle_capacity)
    for m in range(mobile_locker_num):
        model.capacity.add(
            expr=sum(model.O[i, u, m]*demand[i] for i in range(num_node) for u in range(len(parking_pot))) <= mobile_locker_capa)
    
    ## (5)所有车辆或移动自提柜都从仓库出发并返回仓库
    model.start_back_depot = pyo.ConstraintList()
    for k in range(vehicle_num):
        # model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == 1)
        model.start_back_depot.add(expr=sum(model.x[j, num_node-1, k] for j in range(num_node-1)) == 1)
        # model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == sum(model.x[j, 0, k] for j in range(1, num_node)))
        model.start_back_depot.add(expr=sum(model.x[0, j, k] for j in range(1, num_node)) == 1)
        # 仓库之间不可以相互走
        model.start_back_depot.add(expr=model.x[0,num_node-1,k] == 0)
        model.start_back_depot.add(expr=model.x[num_node-1,0,k] == 0)
        for i in range(1,num_node-1):
            model.start_back_depot.add(expr=model.x[i,0,k] == 0)
            model.start_back_depot.add(expr=model.x[num_node-1,i,k] == 0)
    
    for m in range(mobile_locker_num):
        model.start_back_depot.add(expr = sum(model.z[v, len(parking_pot)-1, m] for v in range(len(parking_pot)-1)) == 1)
        model.start_back_depot.add(expr = sum(model.z[0, v, m] for v in range(1, len(parking_pot))) ==1)
        # 仓库之间不可以相互走
        model.start_back_depot.add(expr=model.z[0,len(parking_pot)-1,m] == 0)
        model.start_back_depot.add(expr=model.z[len(parking_pot)-1,0,m] == 0)
        for u in range(1, len(parking_pot)-1):
            model.start_back_depot.add(expr=model.z[u,0,m] == 0)
            model.start_back_depot.add(expr=model.z[len(parking_pot)-1,u,m] == 0)
            

    ## (6)车辆和移动自提柜负载平衡约束
    # model.capacity_balance = pyo.ConstraintList()
    # for j in range(num_node):
    #     for k in range(vehicle_num):
    #         model.capacity_balance.add(
    #             expr=sum(model.r[i, j, k] for i in range(num_node)) - demand[j] * model.y[j, k] == sum(
    #                 model.r[j, i, k] for i in range(num_node)))
    # for v in range(len(parking_pot)):
    #     for m in range(mobile_locker_num):
    #         model.capacity_balance.add(expr=sum(model.g[u, v, m] for u in range(len(parking_pot))) - sum(
    #             demand[i] * model.O[i, u, m] for i in range(num_node) for u in range(len(parking_pot))) == sum(
    #             model.g[v, u, m] for u in range(len(parking_pot))))

    
    # depot约束
    model.depot_cons = pyo.ConstraintList()
    for m in range(mobile_locker_num):
        for i in range(num_node):
            model.depot_cons.add(expr=model.O[i,0,m] == 0)
            model.depot_cons.add(expr=model.O[i,len(parking_pot)-1,m] == 0)
        for u in range(len(parking_pot)):
            model.depot_cons.add(expr=model.O[0,u,m] == 0)
            model.depot_cons.add(expr=model.O[num_node-1,u,m] == 0)

    # (7)顾客自提距离约束
    model.customer_locker_dist = pyo.ConstraintList()
    for i in range(1,num_node-1):
        for m in range(mobile_locker_num):     
            for u in range(1,len(parking_pot)-1):
                model.customer_locker_dist.add(expr=dist_to_parking[i, u] - 100000 * (1 - model.O[i, u, m]) <= L)

    ## (8)候选点数量约束
    # model.ml_num_cons = pyo.ConstraintList()
    # model.ml_num_cons.add(expr=sum(
    #     model.z[u, v, m] for u in range(len(parking_pot)) for v in range(len(parking_pot)) for m in
    #     range(mobile_locker_num)) - sum(
    #     model.z[0, v, m] for v in range(len(parking_pot)) for m in range(mobile_locker_num)) <= len(parking_pot) - 1)

    # (8)每个候选自提点最多仅能访问一次
    model.ml_num_cons = pyo.ConstraintList()
    for v in range(1, len(parking_pot)-1):
        model.ml_num_cons.add(expr=sum(model.z[u,v,m] for u in range(len(parking_pot)-1) for m in range(mobile_locker_num) if u!=v) <= 1)
        
    

    ## (9)送货上门的客户节点必须有车辆进入或离开
    # model.vehicle_to_cust_cons = pyo.ConstraintList()
    # for j in range(1, num_node):
    #     for k in range(vehicle_num):
    #         model.vehicle_to_cust_cons.add(expr=sum(model.x[i, j, k] for i in range(num_node)) == model.y[j, k])
    #         model.vehicle_to_cust_cons.add(expr=sum(model.x[j, i, k] for i in range(num_node)) == model.y[j, k])

    ## (10)顾客自提对应的候选点必须被移动自提柜访问
    model.ml_to_cust_cons = pyo.ConstraintList()
    for u in range(1, len(parking_pot)-1):
        for i in range(1,num_node-1):
            for m in range(mobile_locker_num):
                model.ml_to_cust_cons.add(expr=sum(model.z[v,u,m] for v in range(len(parking_pot)-1) if u!=v) >= model.O[i,u,m])
                # model.ml_to_cust_cons.add(expr=sum(model.z[u,v,m] for v in range(len(parking_pot)-1) if u!=v) <= model.O[i,u,m])
                model.ml_to_cust_cons.add(expr=sum(model.z[v,u,m] for v in range(len(parking_pot)-1) if u!=v) <= sum(model.O[i,u,m] for i in range(1,num_node-1)))
                
    ## 车辆和自提柜到达下一点的时间约束
    # 车辆从仓库出发的时间都为0
    model.arrive_time = pyo.ConstraintList()
    for k in range(vehicle_num):
        model.arrive_time.add(expr=model.ta[0,k] == 0)
    
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                continue
            for k in range(vehicle_num):
                model.arrive_time.add(expr=model.ta[j, k] >=  model.ta[i, k] + model.tw[i, k] + service_time[i] + (dist_mat[i][j] / car_v) - 100000*(1-model.x[i,j,k]))
    
    # 时间窗约束
    model.arrive_due_time = pyo.ConstraintList()
    for k in range(vehicle_num):
        for i in range(num_node):
            model.arrive_due_time.add(expr=model.ta[i,k] <= time_window[i][1])
    
    # 自提柜从仓库出发的时间都为0
    for m in range(mobile_locker_num):
        model.arrive_time.add(expr=model.tta[0,m] == 0)
    
    for u in range(len(parking_pot)):
        for v in range(len(parking_pot)):
            if u == v:
                continue
            
            for m in range(mobile_locker_num):
                model.arrive_time.add(
                    expr=model.tta[v, m] >= model.ttl[u, m] + (parking_dist_mat[u][v] / mobile_locker_v)-100000*(1-model.z[u,v,m]))
    
    
    # # 服务时间约束，该条件只能任意成立一个，所以不能这么写
    model.service_time_cons = pyo.ConstraintList()
    for i in range(1,num_node-1):
        for j in range(1,num_node-1):
            if i!=j:
                for u in range(1,len(parking_pot)-1):
                    for m in range(mobile_locker_num):
                        if time_window[j][0] >= time_window[i][0]:
                            model.service_time_cons.add(expr=model.ts[j, u, m] - model.ts[i, u, m] + 100000 *(1-model.O[j,u,m]) + 100000 *(1-model.O[i,u,m]) >= service_time[i])
                        else:
                            model.service_time_cons.add(expr=model.ts[j, u, m] - model.ts[i, u, m] - 100000 *(1-model.O[i,u,m]) - 100000 *(1-model.O[j,u,m]) <= -service_time[i])

    # # 开始时间约束
    model.start_time_cons = pyo.ConstraintList()
    for i in range(num_node):
        for u in range(len(parking_pot)):
            for m in range(mobile_locker_num):
                model.start_time_cons.add(expr=model.ts[i, u, m] <= model.O[i,u,m] * time_window[i][1])
                model.start_time_cons.add(expr=model.ts[i, u, m] >= model.O[i,u,m] * time_window[i][0])
                model.start_time_cons.add(
                    expr=model.tta[u, m] <= model.ts[i, u, m] + 100000 * (1 - model.O[i, u, m]))  # 自提柜的到达时间约束
                model.start_time_cons.add(expr=model.ttl[u, m] >= model.ts[i, u, m] + service_time[i] - 100000 * (1 - model.O[i, u, m]))  # 自提柜的离开时间约束

    # 车辆等待时间约束
    model.waiting_time_cons = pyo.ConstraintList()
    for i in range(num_node):
        for k in range(vehicle_num):
            # model.waiting_time_cons.add(expr=model.tw[i, k] >= 0)
            model.waiting_time_cons.add(expr=model.tw[i, k] >= time_window[i][0] - model.ta[i, k])
            
    # 移动自提柜闲置时间约束
    for u in range(len(parking_pot)):
        for m in range(mobile_locker_num):
            model.waiting_time_cons.add(expr=model.ttw[u,m] >= model.ttl[u,m]-model.tta[u,m] - sum(model.O[i,u,m] * service_time[i] for i in range(num_node)))
            

    return model


# 求解器设置
def calObjFromSolvers(model, mtype, tmlimit=21600):    # 默认6个小时
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
    # file_name = "rc101"
    # file_name = file_name[:2]
    file_name = ["c101", "c105", "c107", "c109",
                 "r105", "r109", "r110",
                 "r201", "r208", "r211",
                 "rc101", "rc106", "rc107",
                 "rc201", "rc205", "rc208"]
    # file_name = file_name[4:]
    global data_used, vehicle_num, vehicle_capacity, car_v, dist_mat, data_x_y, parking_pot, parking_dist_mat, dist_to_parking, demand, mobile_locker_num, mobile_locker_capa, mobile_locker_v, num_node, service_time, time_window
    all_in = True
    heu = True
    carrr_num = 12 # 车辆数
    # 写个多进程跑一下
    if all_in:
        time_start = time.time()
        res_all = []
        for f in file_name:
            t1 = time.time()
            try:
                f_name = f + ".txt"
                if heu:
                    carrr_num = np.random.randint(15,20)
                
                data_used, vehicle_num, vehicle_capacity, car_v, dist_mat, data_x_y, parking_pot, parking_dist_mat, dist_to_parking, demand = get_point_data(
                    file_path, f_name, 51, 6, carrr_num, True)
                service_time = data_used['service_time'].tolist()
                num_node = len(dist_mat)
                time_window = [(data_used['tw_early'].iloc[i], data_used['tw_late'].iloc[i]) for i in range(num_node)]
                mobile_locker_num = 7  # 移动自提柜的数量
                mobile_locker_capa = 300  # 移动自提柜的容量
                mobile_locker_v = 1
                L = 10
                alpha = 1000
                beta = 1200
            
                model = setModel(L, alpha, beta)
                # 保存模型
                # f = open(r'..\MobileLocker\model.txt', 'w')
                # sys.stdout = f
                # model.pprint()
                # f.close()
                
                calObjFromSolvers(model, 'grb', 600)  # 6h
                print('Min Cost:', model.tot_cost())
                
                z = []
                tot_ml_dis = 0
                ml_used = []
                for u in range(len(parking_pot)):
                    for v in range(len(parking_pot)):
                        for m in range(mobile_locker_num):
                            if pyo.value(model.z[u,v,m]) == 1:
                                tot_ml_dis += parking_dist_mat[u][v]
                                z.append((u,v,m,pyo.value(model.z[u,v,m])))
                                if m not in ml_used:
                                    ml_used.append(m)

                x = []
                tot_car_dis = 0
                car_used = []
                for i in range(num_node):
                    for j in range(num_node):
                        for k in range(vehicle_num):
                            if pyo.value(model.x[i,j,k]) == 1:
                                tot_car_dis += dist_mat[i][j]
                                x.append((i,j,k,pyo.value(model.x[i,j,k])))
                                if k not in car_used:
                                    car_used.append(k)
                                
                O = []
                parking_used = []
                for i in range(num_node):
                    for u in range(len(parking_pot)):
                        for m in range(mobile_locker_num):
                            if pyo.value(model.O[i,u,m]) == 1:
                                O.append((i,u,m,pyo.value(model.O[i,u,m])))    
                                if u not in parking_used:
                                    parking_used.append(u)
                
                car_w = []
                tot_car_w = 0
                for i in range(num_node):
                    for k in range(vehicle_num):
                        if pyo.value(model.tw[i,k]) > 0:
                            tot_car_w += pyo.value(model.tw[i,k])
                            car_w.append((i,k,pyo.value(model.tw[i,k])))
                
                locker_w = []
                tot_ml_w = 0
                for u in range(len(parking_pot)):
                    for m in range(mobile_locker_num):
                        if pyo.value(model.ttw[u,m]) != 0:
                            tot_ml_w += pyo.value(model.ttw[u,m])
                            locker_w.append((u,m,pyo.value(model.ttw[u,m])))
                
                ts = []
                for i in range(num_node):
                    for u in range(len(parking_pot)):
                        for m in range(mobile_locker_num):
                            if pyo.value(model.ts[i,u,m]) > 0:
                                ts.append((i,u,m,pyo.value(model.ts[i,u,m]))) 
                ttl = []
                for u in range(len(parking_pot)):
                    for m in range(mobile_locker_num):
                        if pyo.value(model.ttl[u,m]) > 0:
                            ttl.append((u,m,pyo.value(model.ttl[u,m]))) 
                
                tta = []
                for u in range(len(parking_pot)):
                    for m in range(mobile_locker_num):
                        if pyo.value(model.tta[u,m]) > 0:
                            tta.append((u,m,pyo.value(model.tta[u,m])))
                res_all.append([f, model.tot_cost(),tot_car_dis,len(car_used),tot_ml_dis,len(ml_used),tot_car_w,tot_ml_w,len(parking_used),time.time() - t1, alpha, beta, L])
            except:
                continue
                                     
        ress = pd.DataFrame(res_all,
                        columns=['data_name','best solution', 'vehicle distance', 'vehicle num', 'ml distance', 'ml num','vehicle wait time',
                                 'pentalty time', 'park used','time_limit','alpha','beta','L'])
        ress.to_excel(r'..\MobileLocker\solver_50_1000_1200_6_10_grb_result.xlsx', index=False)
        print("所有数据跑完用时：", time.time() - time_start)
