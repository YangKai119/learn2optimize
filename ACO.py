
import numpy as np
import matplotlib.pyplot as plt

# 超参数
N = 20            # 蚁群数
max_iter = 1000    # 最大迭代次数
rho = 0.9         # 信息素挥发系数
p0 = 0.2          # 转移概率常数
step = 0.1   # 局部搜索步长
tau = []     # 信息素矩阵
X = []       # 种群搜索解
trace = []   # 搜索轨迹

def get_fitness(x):
    return 2*np.sin(x)+np.cos(x)

for i in range(N):
    x = np.random.uniform()
    X.append(x)
    tau.append(get_fitness(x))

for nc in range(1, max_iter + 1):
    lamda = 1 / nc
    tau_best = max(tau)
    bestIdx = tau.index(tau_best)
    p = []
    for i in range(N):
        pi = (tau[bestIdx] - tau[i]) / tau[bestIdx]    # 转移概率
        p.append(pi)
    for i in range(N):
        if p[i] < p0:
            tmp = X[i] + (2 * np.random.uniform() - 1) * step * lamda  # 朝着特定方向走
        else:
            tmp = X[i] + (np.random.uniform() - 0.5)   # 随机走
        if get_fitness(tmp) > get_fitness(X[i]):
            X[i] = tmp
    for i in range(N):
        tau[i] = (1 - rho) * tau[i] + get_fitness(X[i])
    val = max(tau)
    idx = tau.index(val)
    trace.append(get_fitness(X[idx]))

maxVal = max(tau)
maxIdx = tau.index(maxVal)
print(maxVal)

# 画适应度变化曲线
x = np.array([x for x in range(max_iter)])
y = np.array(trace)
plt.plot(x,y)
plt.show()













