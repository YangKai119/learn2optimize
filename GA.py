
import numpy as np
import matplotlib.pyplot as plt

# 适应度函数
def get_fitness(x):
    return 3*(1-x)**2*np.exp(-(x+1)**2)- 10*(x/5 - x**3)*np.exp(-x**2)- 1/3**np.exp(-(x+1)**2)

# 种群类
class inds():
    def __init__(self):
        self.x = 0   # 解
        self.fitness = 0  # 适应度
    def __eq__(self,other):
        self.x = other.x
        self.fitness = other.fitness
# 初始化种群
def init(pop,N):
    for i in range(N):
        ind = inds()
        ind.x = np.random.uniform(0,5)
        ind.fitness = get_fitness(ind.x)
        pop.append(ind)
# 随机选择算子
def select(N):
    return np.random.choice(N,2)
# 交叉算子（连续编码）
def crossover(par1,par2):
    child1,child2 = inds(),inds()
    child1.x = 0.9*par1.x + 0.1*par2.x
    child2.x = 0.1*par1.x + 0.9*par2.x
    child1.fitness = get_fitness(child1.x)
    child2.fitness = get_fitness(child2.x)
    return child1,child2
# 变异算子
def muta(pop):
    ind = np.random.choice(pop)
    ind.x = np.random.uniform(0,5)
    ind.fitness = get_fitness(ind.x)
    return ind
# 超参数
N = 20
POP = []
max_iter = 500
# 初始化种群
init(POP,N)
fitness = []
# 迭代开始
for i in range(max_iter):
    a,b = select(N)
    # 交叉
    if np.random.rand() < 0.75:
        child1,child2 = crossover(POP[a],POP[b])
        new = sorted([POP[a],POP[b],child1,child2],
                     key = lambda ind:ind.fitness,
                     reverse = True)
        POP[a],POP[b] = new[0],new[1]
    # 变异
    if np.random.rand() < 0.1:
        muta(POP)

    POP.sort(key=lambda ind:ind.fitness, reverse=True)
    fitness.append(POP[0].fitness)

print(fitness[-1])

# 画适应度变化曲线
x = np.array([x for x in range(max_iter)])
y = np.array(fitness)
plt.plot(x,y)
plt.show()







