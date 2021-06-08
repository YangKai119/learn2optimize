import json
from GapModel import *
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
    path = 'D:/办公文件/研究生项目/指派问题/GAP_master/data/'
    train_filename = "gap_15_10_train.json"
    eval_filename = "gap_15_10_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = GapModel(data)
    init_sol = model.get_random_init_solution()

    print("random初始解：",init_sol.tot_benfit)
    print(init_sol.sol_seq)
    PSO = ParticleSwarmOptimization(init_sol)
    sol = PSO.solver()
    obj_plot(sol.fitness)
    print("PSO优化解：", sol.tot_benfit)
    print("算法运行时间：", time.time() - time_start)

