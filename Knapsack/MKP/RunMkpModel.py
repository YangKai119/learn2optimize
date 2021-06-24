import json
from MkpModel import *
from OptAlgorithm import *
import matplotlib.pyplot as plt
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
    path = '../data/mkp/'
    train_filename = "mkp_10_300_400_train.json"
    eval_filename = "mkp_10_300_400_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = MkpModel(data)
    init_sol = model.parse()

    DE = DifferentialEvolution(init_sol)
    sol = DE.solver()
    obj_plot(sol.fitness)
    print("DE初始解：", sol.fitness[0])
    print("DE优化解：", sol.tot_val)
