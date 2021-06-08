"""
1.生成10个工人，15个工作，进行指派
2.优化目标是总收益最大化
3.每位工人并不是所有的工作都能完成
"""

import numpy as np
import json
import os

# W10T15
def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        num_works = 10
        cur_sample['workers'] = num_works
        num_tasks = 15
        cur_sample['tasks'] = num_tasks
        cur_sample['max_tasks_work'] = 3
        cur_sample['nodes'] = []
        for i in range(num_tasks):
            if np.random.rand() > 0.7:    # 生成收益矩阵
                work_benfit = np.random.randint(1,8,num_works)
            else:
                work_benfit = np.random.randint(10,15,num_works)
            # machine_id = np.random.randint(0,10)
            worker_ids = [i for i in range(num_works)]
            if np.random.rand() > 0.88:
                num = np.random.randint(1,5) # 随机生成工人无法完成任务的个数
                not_worker_id = np.random.choice(range(num_works), num, replace=False)  # 不重复抽样
                for idx in not_worker_id:
                    worker_ids.remove(idx)
                    work_benfit[idx] = 0
            work_benfit = list(map(float, work_benfit))
            cur_sample['nodes'].append({'worker_ids': worker_ids,'work_benfit': work_benfit})
        samples.append(cur_sample)

    return samples

if __name__ == "__main__":
    samples = main()
    path = './data/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "gap_15_10.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "gap_15_10_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "gap_15_10_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "gap_15_10_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)
