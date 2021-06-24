
import json
from TwoDimBpModel import *
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
    path = '../data/2Dbp/'
    train_filename = "2dbp_5_300_200_train.json"
    eval_filename = "2dbp_5_300_200_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = BpModel(data)
    init_sol = model.parse()
    IA = ImmuneAlgorithm(init_sol)
    sol = IA.solver()
    print("IA初始解：", sol.init_fit)
    obj_plot(sol.fitness)
    # print(len(sol.bins))
    print("IA优化解：", sol.best_fit)
