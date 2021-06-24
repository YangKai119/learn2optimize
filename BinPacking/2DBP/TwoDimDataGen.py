"""
考虑二维装箱问题
目标函数为最小化使用箱子的个数
只考虑面积约束
"""

import numpy as np
import json
import os


def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        cur_sample['goods'] = []
        cur_sample['bin_length'] = 300    # 单位cm
        cur_sample['bin_width'] = 200
        num_goods = 5
        for i in range(num_goods):
            if np.random.uniform() > 0.5:    # 大件物品
                length = np.random.randint(40, 50)    # 单位cm
                width = np.random.randint(20, 30)
                num = np.random.randint(50, 100)
            else:                           # 小件物品
                length = np.random.randint(20, 30)
                width = np.random.randint(10, 20)
                num = np.random.randint(100, 150)
            cur_sample['goods'].append({'length': length, 'width': width,  'num': num})
        samples.append(cur_sample)
    return samples

if __name__ == "__main__":
    samples = main()
    path = '../data/2Dbp/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "2dbp_5_300_200.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "2dbp_5_300_200_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "2dbp_5_300_200_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "2dbp_5_300_200_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)
