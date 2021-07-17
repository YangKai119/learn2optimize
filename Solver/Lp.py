
from pyomo.environ import *
from pyomo.core import Suffix

model = ConcreteModel()
model.dual = Suffix(direction=Suffix.IMPORT)   # 对偶问题约束

model.x1 = Var(domain=NonNegativeReals)
model.x2 = Var(domain=NonNegativeReals)

model.profit = Objective(expr=model.x1*2 + model.x2*3, sense=maximize)

model.equipment = Constraint(expr=model.x1+model.x2*2 <= 8)
model.origin_A = Constraint(expr=model.x1*4 <= 16)
model.origin_B = Constraint(expr=model.x2*4 <= 12)

# 直接把地址导过来就OK了
result = SolverFactory('glpk', executable='D:/办公文件/编程/数学建模/solver/winglpk-4.65/glpk-4.65/w64/glpsol').solve(model)
result.write()

print(model.profit())  # 要加括号
print(model.x1())
print(model.x2())

print(-model.dual[model.equipment])   # 影子价格
print(-model.dual[model.origin_A])
print(-model.dual[model.origin_B])

str = '{0:7.2f} {1:7.2f} {2:7.2f} {3:7.2f}'
print('Constraint value lslack uslack dual')
for c in (model.equipment, model.origin_A, model.origin_B):
    print(c, str.format(c(), c.lslack(), c.uslack(), model.dual[c]))



