import json
from TspModel import *
from OptAlgorithm import *
import matplotlib.pyplot as plt
from RoutesPlot import *
import time

def load_dataset(filename, path):
    with open(path + filename, 'r') as f:
        samples = json.load(f)
    print('Number of data samples in ' + filename + ': ', len(samples))
    return samples

def obj_plot(fitness):    # 画适应度函数图
    x = np.array([x for x in range(len(fitness))])
    y = np.array(fitness)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    path = '../data/tsp/'
    train_filename = "tsp_50_500_train.json"
    eval_filename = "tsp_50_500_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = TspModel(data)
    init_sol = model.get_greedy_init_solution()
    SA = SimulatedAnnealing(init_sol)
    # TS = TabuSearch(init_sol)
    print("greedy初始解：",init_sol.tot_dist)
    sol = SA.solver()
    DrawPointMap(sol, 'SA')
    obj_plot(sol.fitness)
    print("SA优化解：",sol.tot_dist)
    print("算法运行时间：",time.time()-time_start)
