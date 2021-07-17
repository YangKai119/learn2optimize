
import pyomo.environ as pye
import pandas as pd
from pyomo.core import TransformationFactory
from pyomo.gdp import Disjunction
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np

def exp1():
    data = {}
    for i in range(4):
        data[i] = [np.random.randint(1,10) for _ in range(4)]
    df = pd.DataFrame(data,index = ['0','1','2','3'])
    print(df)

    model = pye.ConcreteModel()
    model.x = pye.Var(df.index, df.columns, domain=pye.Binary)

    def _cost_sum():
        return sum(df.loc[i] * model.x[i] for i in model.x)

    model.cost = pye.Objective(expr=_cost_sum())

    model.mission = pye.Constraint(df.columns, rule=lambda m, j: sum(m.x[i,j] for i in df.index) == 1)
    model.person = pye.Constraint(df.index, rule=lambda m, i: sum(m.x[i, j] for j in df.columns) == 1)

    opt = pye.SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol', name='TMA')
    result = opt.solve(model,tee=False)

    out = pd.DataFrame(0,index=df.index,columns=df.columns)
    for i in model.x:
        out.loc[i] = pye.value(model.x[i])
    print(out)

def exp2():
    data = {}
    for i in range(4):
        data[i] = [np.random.randint(1, 10) for _ in range(5)]
    df = pd.DataFrame(data, index=['0', '1', '2', '3','4'])
    print(df)

    model = pye.ConcreteModel()
    model.x = pye.Var(df.index, df.columns, domain=pye.Binary)

    def _cost_sum():
        return sum(df.loc[i] * model.x[i] for i in model.x)

    model.cost = pye.Objective(expr=_cost_sum())

    model.mission = pye.Constraint(df.columns, rule=lambda m, j: sum(m.x[i, j] for i in df.index) == 1)
    model.person = pye.Constraint(df.index, rule=lambda m, i: sum(m.x[i, j] for j in df.columns) <= 1)

    excl_pairs = [('1','2'),('1','4')]   # 作业冲突，1与2，或者1与4都不能同时干活
    model.disj = Disjunction(excl_pairs, rule=lambda m,a,b: [sum(model.x[a,j] for j in df.columns) == 0,
                                                             sum(model.x[b,j] for j in df.columns) == 0])
    TransformationFactory('gdp.hull').apply_to(model)
    opt = pye.SolverFactory('glpk', executable='../solver/winglpk-4.65/glpk-4.65/w64/glpsol', name='TMA')
    result = opt.solve(model, tee=True)   # 参数tee为打印计算过程

    out = pd.DataFrame(0, index=df.index, columns=df.columns)
    for i in model.x:
        out.loc[i] = pye.value(model.x[i])
    print(out)
    print(model.cost())


if __name__ == '__main__':
    exp2()






