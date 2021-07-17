'''
运输问题
--表上作业法
（运输计划）
产销平衡问题
产销不平衡问题
'''

from pyomo.environ import *
import pandas as pd
import numpy as np

def exp1():
    factory = {'A1':7,'A2':4,'A3':9}   # 工厂产量
    market = {'B1':3,'B2':6,'B3':5,'B4':6}   # 市场需求
    # for x in factory:   只打印key
    #     print(x)
    cost_price = {}   # 运输成本
    for a in factory.keys():
        for b in market.keys():
            cost_price[(a,b)] = np.random.randint(1,10)

    df = pd.DataFrame(0, index=factory, columns=market)
    for (x,y),v in cost_price.items():
        df.loc[x,y]=v
    print(df)

    # 建模
    model = ConcreteModel()
    model.x = Var(factory,market,domain=NonNegativeReals)
    # x,y为参数变量索引
    model.cost = Objective(expr=sum(model.x[x,y]*cost_price[(x,y)] for x in factory.keys() for y in market.keys()),
                           sense=minimize)

    model.supply = ConstraintList()
    for f,v in factory.items():
        model.supply.add(expr=sum(model.x[f,m] for m in market.keys()) == v)

    model.demand = ConstraintList()
    for m,v in market.items():
        model.demand.add(expr=sum(model.x[f,m] for f in factory.keys()) == v)

    print(model.x._data)

    # 直接把地址导过来就OK了
    result = SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol').solve(model)
    result.write()

    print('Min cost:',model.cost())
    print('Result:')
    df = pd.DataFrame(0, index=factory,columns=market)
    for x,y in cost_price:
        df.loc[x,y] = model.x[x,y]()

    print(df)

def exp2():
    fee = pd.DataFrame({'e':[16,14,19],'f':[13,13,20],'g':[22,19,23],'h':[17,15,9999]},
                       index=['A','B','C'])
    supply = {'A':50,'B':60,'C':50}
    demand = {'e':{'low':30,'up':50},
              'f':70,
              'g':{'low':0,'up':30},
              'h':{'low':10}}
    print(fee)

    model = ConcreteModel()
    valid_key = set((i,j) for i in supply for j in demand if not (i == 'C' and j == 'h'))   # 意思是除了C->h其他的都是有效的变量索引
    model.x = Var(valid_key,domain=NonNegativeReals)
    # 可以随便在模型中起名字
    model.cost = Objective(expr=sum(model.x[i,j]*fee.loc[i,j] for i,j in valid_key),
                           sense=minimize)

    model.supply = ConstraintList()
    for i,v in supply.items():
        model.supply.add(expr=sum(model.x[i,j] for j in demand if (i,j) in valid_key) == v)

    model.demand = ConstraintList()
    for j,v in demand.items():
        if isinstance(v,dict):
            if 'low' in v:
                model.demand.add(expr=sum(model.x[i,j] for i in supply if (i,j) in valid_key) >= v['low'])
            if 'up' in v:
                model.demand.add(expr=sum(model.x[i,j] for i in supply if (i,j) in valid_key) <= v['up'])
        else:
            model.demand.add(expr=sum(model.x[i, j] for i in supply if (i, j) in valid_key) == v)

    # 直接把地址导过来就OK了
    result = SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol').solve(model)
    result.write()

    print('Min cost:', model.cost())
    schedule = pd.DataFrame(0, index=fee.index, columns=fee.columns)
    for i,j in valid_key:
        schedule.loc[i,j] = model.x[i,j]()
    print(schedule)

def exp3():
    supply = {1:25,2:35,3:30,4:10}
    supply_cost = {1:10.8,2:11.1,3:11.0,4:11.3}
    persistence_cost = 0.15
    demand = {1:10,2:15,3:25,4:20}

    model = ConcreteModel()
    valid_key = set()
    for i in supply:
        for j in demand:
            if i <= j:
                valid_key.add((i,j))

    model.x = Var(valid_key, domain=NonNegativeReals)

    def fee(i,j):
        return round(supply_cost[i] + (j-i)*persistence_cost, 2)

    model.cost = Objective(expr=sum(model.x[i,j]*fee(i,j) for i,j in valid_key))

    model.supply = ConstraintList()
    for i,v in supply.items():
        model.supply.add(expr=sum(model.x[i,j] for j in demand if (i,j) in valid_key) <= v)

    model.demand = ConstraintList()
    for j, v in demand.items():
        model.demand.add(expr=sum(model.x[i, j] for i in supply if i <= j) == v)

    # 直接把地址导过来就OK了
    result = SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol').solve(
        model)
    result.write()

    print('Min cost:', model.cost())

if __name__ == '__main__':
    exp2()
