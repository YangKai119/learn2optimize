
import numpy as np
import matplotlib.pyplot as plt

L = np.random.randint(5,11)            # 禁忌长度取5,11之间的随机数
Ca = 5                                       # 邻域解个数
max_iter = 1000                             # 禁忌算法的最大迭代次数
w = 1                                       # 自适应权重系数
tabu = []                                      # 禁忌表
x0 = np.random.uniform()                  # 随机产生初始解
trace = [x0]                                   # 每次迭代最优解的轨迹

# 一个解的类
class Sol():
    def __init__(self):
        self.x = 0
        self.val = 0
    def __eq__(self, other):
        self.x = other.x
        self.val = other.val

def get_fitness(x):
    return -x**4 + 4*x**5

best_sol = Sol()
xnow = [Sol()] * max_iter
best_sol.x = x0          # 最优解
best_sol.val = get_fitness(best_sol.x)
xnow[0].x = x0           # 当前解
xnow[0].val = get_fitness(xnow[0].x)
candidate = [Sol()] * max_iter  # 每次迭代的候选解

g = 0
while g < max_iter-1:
    x_near = []    # 领域解
    w = w * 0.998
    fitval_near = []
    for i in range(Ca):
        # 产生邻域解
        x = xnow[g].x
        x_temp = x + (2 * np.random.uniform() - 1) * w
        x_near.append(x_temp)
        fitval_near.append(get_fitness(x_temp))     # 计算邻域解点的函数值

    near_best_fit = max(fitval_near)
    bestIdx = fitval_near.index(near_best_fit)   # 最优邻域解为候选解
    candidate[g].x = x_near[bestIdx]
    candidate[g].val = get_fitness(candidate[g].x)
    delta1 = candidate[g].val-xnow[g].val   # 候选解和当前解的评价函数差
    delta2 = candidate[g].val-best_sol.val  # 候选解和目前最优解的评价函数差

    # 候选解并没有改进解，把候选解赋给下一次迭代的当前解
    if delta1 <= 0:
        xnow[g+1].x = candidate[g].x
        xnow[g+1].val = get_fitness(xnow[g].x)
        tabu.append(xnow[g+1].x)    # 更新禁忌表
        if len(tabu) > L:
            tabu = []
        g += 1                 # 更新禁忌表后，迭代次数自增1

    else:
        if delta2 > 0:            # 候选解比目前最优解优
            xnow[g+1].x = candidate[g].x    # 把改进解赋给下一次迭代的当前解
            xnow[g+1].val = get_fitness(xnow[g+1].x)
            tabu.append(xnow[g+1].x)
            if len(tabu) > L:
                tabu = []
            best_sol.x = candidate[g].x    # 把改进解赋给下一次迭代的目前最优解
            best_sol.val = get_fitness(best_sol.x)   # 包含藐视准则
            g += 1

        else:
            # 判断改进解时候在禁忌表里
            r = 0
            if candidate[g].x in tabu:
                r = 1
            if r == 0:
                xnow[g+1].x = candidate[g].x
                xnow[g+1].val = get_fitness(xnow[g+1].x)
                tabu.append(xnow[g].x)
                if len(tabu) > L:
                    tabu = []
                g += 1
            else:                    # 如果改进解在禁忌表里，用当前解重新产生邻域解
                xnow[g].x = xnow[g].x
                xnow[g].val = get_fitness(xnow[g].x)

    trace.append(get_fitness(best_sol.x))

print(trace[-1])

# 画适应度变化曲线
x = np.array([x for x in range(max_iter)])
y = np.array(trace)
plt.plot(x,y)
plt.show()





