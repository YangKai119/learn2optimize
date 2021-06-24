
import json
from ThreeDimBpModel import *
from OptAlgorithm import *
import matplotlib.pyplot as plt
from BinBoxPlot import *
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

def bin_box_plot(sol):
    O = (0, 0, 0)  # 原点坐标
    C = (sol.bin_length, sol.bin_width, sol.bin_height)  # 箱体长宽高
    color = 'red'  # 箱体颜色
    # 显示箱体
    # show_pic([packaging(O, C, color)])
    for i in range(len(sol.bins)):
        bin = sol.bins[i]
        show_num = [packaging(O, C, color)]  # 这个为后面组合显示时，把箱体显示数据添加到所有要显示的数据里面
        color2 = 'blue'
        for idx in range(len(bin.goods_seq)):
            goods = bin.goods_seq[idx]
            x,y,z = sol.decode_direction(goods,goods.direction)
            length_width_heigth = (x,y,z)
            set_point = goods.set_point   # 有重叠的放置点
            show_num.append(packaging(set_point, length_width_heigth, color2))
        show_pic(show_num)   # 这个为直接显示箱体看下效果

if __name__ == "__main__":
    path = '../data/3Dbp/'
    train_filename = "3dbp_5_300_200_150_train.json"
    eval_filename = "3dbp_5_300_200_150_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    time_start = time.time()
    data = train_data[0]
    model = BpModel(data)
    init_sol = model.parse()
    GA = GeneticAlgorithm(init_sol)
    sol = GA.solver()
    print("GA初始解：", sol.init_fit)
    obj_plot(sol.fitness)
    # print(len(sol.bins))
    bin_box_plot(sol)
    print("GA优化解：", sol.best_fit)
