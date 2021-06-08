
"""
1.生成j个工件p道工序m台机器
2.每道工序对应的机器并不唯一
3.是否需要考虑机器资源的约束？
4.每道工序都有其自身的工作时间
5.前一道工序未结束时，下一道工序无法开始
6.所有机器同时开工
"""

import numpy as np
import json
import os

# J20P10M10的数据
def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        num_machine = 10
        cur_sample['machines'] = num_machine
        num_jobs = 20
        cur_sample['jobs'] = num_jobs
        num_process = 10
        cur_sample['process'] = num_process
        cur_sample['nodes'] = []
        for i in range(num_jobs):
            process = []
            for j in range(num_process):
                if np.random.rand() > 0.7:
                    work_time = np.random.randint(1,8,num_machine)
                else:
                    work_time = np.random.randint(10,15,num_machine)
                # machine_id = np.random.randint(0,10)
                machine_ids = [i for i in range(num_machine)]
                if np.random.rand() > 0.88:
                    num = np.random.randint(1,7) # 随机生成机器无法开工的个数
                    not_machine_id = np.random.choice(range(num_machine), num, replace=False)  # 不重复抽样
                    for idx in not_machine_id:
                        machine_ids.remove(idx)
                        work_time[idx] = 9999
                work_time = list(map(float,work_time))  # 转成Float类型
                process.append({'process_idx': j, 'machines_ava': machine_ids, 'work_time': work_time})
            cur_sample['nodes'].append({'process': process})
        samples.append(cur_sample)

    return samples

if __name__ == "__main__":
    samples = main()
    path = 'D:/办公文件/研究生项目/作业调度研究/FJSP-master/FJSP-master/data/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "fjsp_20_10_10.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "fjsp_20_10_10_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "fjsp_20_10_10_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "fjsp_20_10_10_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)