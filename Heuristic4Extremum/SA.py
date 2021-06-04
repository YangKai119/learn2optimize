import numpy as np
import matplotlib.pyplot as plt

# 超参数设置
T = 1e2    # 初始温度
T_min = 1e-3  # 降低到该温度后停止迭代
a = 0.98   # 退火率
b = 1  # 变化参数

def get_fitness(x):
    return -x**4 + 4*x**5

def judge(de,T):
    if de > 0:
        return 1
    else:
        if np.exp(de/T) > np.random.rand():
            return 1
        else:
            return 0

# 初始化参数
x = np.random.rand()
f = get_fitness(x)
max_iter = 0
fitness = []

# 退火过程
while T > T_min:
    x_new = x + (np.random.rand() - 0.5) * b    # 解的更新
    f_new = get_fitness(x_new)
    de = f_new - f
    if judge(de,T):  # 判断是否更新当前解
        f = f_new
        x = x_new
    if de > 0:
        T = T * a
    max_iter += 1
    fitness.append(f)

print(f)

# 画适应度变化曲线
x = np.array([x for x in range(max_iter)])
y = np.array(fitness)
plt.plot(x,y)
plt.show()













