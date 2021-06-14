
import numpy as np
import json
import os

def sample_pos():
    return np.random.rand(), np.random.rand()

def main():
    samples = []
    for _ in range(10000):
        cur_sample = {}
        cur_sample['nodes'] = []
        cur_sample['capacity'] = 500
        num_customers = 50
        for i in range(num_customers+1):
            cx, cy = sample_pos()
            demand = np.random.randint(1, 9) if i != 0 else 0
            cur_sample['nodes'].append({'position': (cx, cy), 'demand': demand})
        samples.append(cur_sample)
    return samples

if __name__ == "__main__":
    samples = main()
    path = '../data/tsp/'
    if not os.path.exists(path):
        os.makedirs(path)

    data_size = len(samples)
    print(data_size)
    res_file = "tsp_50_500.json"
    fout_res = open(path+res_file, 'w')
    json.dump(samples, fout_res)

    res_train_file = "tsp_50_500_train.json"
    fout_train = open(path+res_train_file, 'w')
    train_data_size = int(data_size * 0.8)
    json.dump(samples[:train_data_size], fout_train)

    res_eval_file = "tsp_50_500_eval.json"
    fout_eval = open(path+res_eval_file, 'w')
    eval_data_size = int(data_size * 0.9) - train_data_size
    json.dump(samples[train_data_size: train_data_size + eval_data_size], fout_eval)

    res_test_file = "tsp_50_500_test.json"
    fout_test = open(path+res_test_file, 'w')
    test_data_size = data_size - train_data_size - eval_data_size
    json.dump(samples[train_data_size + eval_data_size:], fout_test)
