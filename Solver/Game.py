
import numpy as np
import pyomo.environ as pye



def main(matrix):
    print('payoff game matrix:')
    print(matrix)
    model = pye.ConcreteModel()
    model.dual = pye.Suffix(direction=pye.Suffix.IMPORT)
    index = list(range(matrix.shape[0]))
    model.x = pye.Var(index, domain=pye.NonNegativeReals)

    model.obj = pye.Objective(expr=sum(model.x[i] for i in index), sense=pye.minimize)
    model.con = pye.ConstraintList()
    for i in range(matrix.shape[1]):
        s = matrix[:,i]
        model.con.add(expr=sum(model.x[i]*v for j,v in enumerate(s)) >= 1)

    solve(model)

    print(model.obj())


def solve(model):
    print("----Solving----")
    opt = pye.SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol', name='GAME')
    results = opt.solve(model,tee=True)

    if results.solver.status == pye.SolverStatus.ok and results.solver.termination_condition == pye.TerminationCondition.optimal:
        print('This is feasible and optimal!')
        return True

    elif results.solver.termination_condition == pye.TerminationCondition.infeasible:
        print('Do something about it? or exit?')
        return False

    else:
        print(str(results.solver))
        raise Exception()


if __name__ == "__main__":
    matrix = np.array([np.random.randint(1,10) for i in range(9)]).reshape(3,3)
    main(matrix)
