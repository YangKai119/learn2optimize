
"""
考虑三维装箱问题
目标函数为最小化使用箱子的个数
不考虑货物打包装箱
"""

import numpy as np
import json
import os


def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        cur_sample['goods'] = []
        cur_sample['bin_capacity'] = 4500   # 载重量，单位kg
        cur_sample['bin_length'] = 300    # 单位cm
        cur_sample['bin_width'] = 200
        cur_sample['bin_height'] = 150
        num_goods = 5
        for i in range(num_goods):
            if np.random.uniform() > 0.5:    # 大件物品
                weight = np.random.randint(4, 9)     # 单位kg
                length = np.random.randint(40, 50)    # 单位cm
                width = np.random.randint(20, 30)
                height = np.random.randint(25, 35)
                num = np.random.randint(50, 100)
            else:                           # 小件物品
                weight = np.random.randint(1, 4)
                length = np.random.randint(20, 30)
                width = np.random.randint(10, 20)
                height = np.random.randint(15, 25)
                num = np.random.randint(100, 150)
            cur_sample['goods'].append({'weight': weight, 'length': length, 'width': width, 'height': height, 'num': num})
        samples.append(cur_sample)
    return samples

if __name__ == "__main__":
    samples = main()
    path = 'D:/办公文件/研究生项目/装箱问题/demo/data/3Dbp/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "3dbp_5_300_200_150.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "3dbp_5_300_200_150_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "3dbp_5_300_200_150_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "3dbp_5_300_200_150_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)







