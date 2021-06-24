
"""考虑10件物品，背包容量20的情况"""



import numpy as np
import json
import os


def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        cur_sample['goods'] = []
        cur_sample['capacity'] = 20
        num_goods = 10
        for i in range(num_goods):
            weight = np.random.randint(1, 9)
            if np.random.uniform() > 0.5:
                value = np.random.randint(10, 15)
            else:
                value = np.random.randint(5, 10)
            cur_sample['goods'].append({'weight': weight, 'value': value})
        samples.append(cur_sample)
    return samples

if __name__ == "__main__":
    samples = main()
    path = '../data/kp/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "kp_10_20.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "kp_10_20_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "kp_10_20_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "kp_10_20_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)
