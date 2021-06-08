
from JspSeq import *
import copy
import numpy as np

class JspModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = JobScheduleManager(self.problem['machines'],self.problem['jobs'],self.problem['process'])
        for i,job in enumerate(self.problem['nodes']):
            process = job['process']
            for idx in range(len(process)):
                sol.jobs_seq[i].append(JobNode(job_id = i,
                                               process_id = idx,
                                               machine_id = process[idx]['machine_id'],
                                               work_time = process[idx]['work_time']))
        for i in range(self.problem['machines']):
            sol.machine_list.append(Machine(i))

        return sol

    def get_random_init_solution(self):
        init_sol = self.parse()
        for job_id in range(init_sol.num_jobs):
            init_sol.sol_seq += [job_id] * init_sol.num_process
        np.random.seed(0)
        np.random.shuffle(init_sol.sol_seq)
        init_sol.cur_obj = init_sol.get_obj()
        # init_sol.best_obj = init_sol.cur_obj
        return init_sol

    def get_greedy_init_solution(self):   # 贪心算法求初始解
        init_sol = self.parse()
        jobs_seq = init_sol.jobs_seq
        all_jobs = []
        for idx in range(len(jobs_seq)):
            all_jobs += jobs_seq[idx]
        # 对所有工作按工序顺序以及工时双排序，工时短的优先进入机器操作
        all_jobs.sort(key = lambda x : (x.process_id, x.work_time))
        for job in all_jobs:
            init_sol.sol_seq.append(job.job_id)
        init_sol.cur_obj = init_sol.get_obj()
        return init_sol









