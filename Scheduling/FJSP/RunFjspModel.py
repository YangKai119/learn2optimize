"""
先读取文件
"""

import json
import time
from FjspModel import *
from OptAlgorithm import *
import matplotlib.pyplot as plt

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

def gantt_chart(sol):    # 画甘特图
    macInfo = []       # 对应的机器号 M1=1,M2=2,…
    flow = []   # 各工序加工时间
    macStartTime = []   # 各工序开始时间
    workpiece = sol.sol_seq[1]    # 工件号 J1=1,J2=2
    operation = []  # 操作序号
    stack = []   # 临时存储的栈
    for job_id in sol.sol_seq[1]:
        process_id = stack.count(job_id)
        stack.append(job_id)  # 多余的内存，后期可以优化
        job_proc = sol.get_process(job_id, process_id)
        macInfo.append(job_proc.machine_id+1)
        flow.append(job_proc.work_time[job_proc.machine_id])
        macStartTime.append(job_proc.start_time)
        operation.append(process_id)

    for j in range(len(macInfo)):
        i = macInfo[j]-1
        plt.barh(i, flow[j], 0.3, left=macStartTime[j]) # 条形图
        plt.text(macStartTime[j] + flow[j] / 10, i, 'J%s.%s' % (workpiece[j], operation[j]), color="white", size=10)

    plt.yticks(np.arange(max(macInfo)), np.arange(1, max(macInfo) + 1))
    plt.show()

if __name__ == "__main__":
    path = 'D:/办公文件/研究生项目/作业调度研究/FJSP-master/JSP_FJSP/data/'
    train_filename = "fjsp_20_10_10_train.json"
    eval_filename = "fjsp_20_10_10_eval.json"
    train_data = load_dataset(train_filename, path)
    eval_data = load_dataset(eval_filename, path)

    data = train_data[0]
    model = FjspModel(data)
    start_time = time.time()
    init_sol = model.get_greedy_init_solution()
    GA = GeneticAlgorithm(init_sol)
    # SA = SimulatedAnnealing(init_sol)
    print("greedy初始解：", init_sol.cur_obj)
    sol = GA.solver()
    obj_plot(sol.fitness)
    gantt_chart(sol)
    print("GA优化解：", sol.cur_obj)
    print("算法运行时间：",time.time() - start_time)






