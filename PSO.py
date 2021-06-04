import numpy as np
import matplotlib.pyplot as plt

# 超参数
w = 0.8
c1 = 1.5
c2 = 1.5
r1 = 0.3
r2 = 0.6
dim = 1  # 适应度函数是几元的就是几维
N = 50
max_iter = 1000

def get_fitness(x):
    return -x**4 + 4*x**5

# 参数初始化
X = np.zeros((N,dim))  # 位置初始化
V = np.zeros((N,dim))  # 速度初始化
p_best = np.zeros((N,dim),dtype=float)  # 当前种群最好位置
p_fit = np.zeros((N,dim),dtype=float)  # 当前最优位置适应度
g_fit = 1e-3   # 初始全局最优解

# 随机初始化
for i in range(N):
    X[i] = np.random.uniform(0,5,dim)
    V[i] = np.random.uniform(0,5,dim)
    p_best[i] = X[i]
    p_fit[i] = get_fitness(X[i])
    if p_fit[i] > g_fit:
        g_fit = p_fit[i]
        g_best = p_best[i]

# 迭代开始
fitness = []
for _ in range(max_iter):
    for i in range(N):
        tmp = get_fitness(X[i])
        if tmp > p_fit[i]:
            p_best[i] = X[i]
            p_fit[i] = tmp
            if p_fit[i] > g_fit:
                g_fit = p_fit[i][:]
                g_best = p_best[i][:]
    for i in range(N):
        V[i] = w * V[i] + c1 * r1 * (p_best[i] - X[i]) + c2 * r2 * (g_best - X[i])  # 速度更新公式
        X[i] = V[i] + X[i]   # 位置更新公式
    fitness.append(g_fit[0])

print(g_fit[0])

# 画适应度变化曲线
x = np.array([x for x in range(max_iter)])
y = np.array(fitness)
plt.plot(x,y)
plt.show()







