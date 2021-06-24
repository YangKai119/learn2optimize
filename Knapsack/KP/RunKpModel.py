
import json
from KpModel import *
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
    path = '../data/kp/'
    train_filename = "kp_10_20_train.json"
    eval_filename = "kp_10_20_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = KpModel(data)
    sol = model.parse()

    # 暴力递归
    # bf = BruteForce(sol)
    # bf.solver()

    # 动态规划
    # dp = DynamicProgramming(sol)
    # dp.solver()

    # 递归回溯
    # bt = BackTracking(sol)
    # bt.solver()

    # 分支界定
    bb = BranchAndBound(sol)
    bb.solver()

    print(sol.tot_obj)
    print(sol.goods_seq)




