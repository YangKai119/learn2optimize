
"""
二维背包问题
考虑10件物品，背包承重、体积约束的情况
二维多重背包问题
仅需要将其转换为0-1背包问题即可
即拓展编码长度为每类物品*各自物品的数量后加总
"""


import numpy as np
import json
import os


def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        cur_sample['goods'] = []
        cur_sample['bag_capacity'] = 400
        cur_sample['bag_volume'] = 300
        num_goods = 10
        for i in range(num_goods):
            weight = np.random.randint(1, 9)
            volume = np.random.randint(1, 7)
            num = np.random.randint(1, 7)    # 加入数量后变为多重背包问题
            if np.random.uniform() > 0.5:
                value = np.random.randint(10, 15)
            else:
                value = np.random.randint(5, 10)
            cur_sample['goods'].append({'weight': weight, 'value': value, 'volume': volume,'num': num})
        samples.append(cur_sample)
    return samples

if __name__ == "__main__":
    samples = main()
    path = '../data/mkp/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "mkp_10_300_400.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "mkp_10_300_400_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "mkp_10_300_400_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "mkp_10_300_400_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)


