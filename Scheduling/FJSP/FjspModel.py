from FjspSeq import *
import copy
import numpy as np

class FjspModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = JobScheduleManager(self.problem['machines'],self.problem['jobs'],self.problem['process'])
        for i,job in enumerate(self.problem['nodes']):
            process = job['process']
            for idx in range(len(process)):
                sol.jobs_seq[i].append(JobNode(job_id = i,
                                               process_id = idx,
                                               machines_ava = process[idx]['machines_ava'],
                                               work_time = process[idx]['work_time']))
        for i in range(self.problem['machines']):
            sol.machine_list.append(Machine(i))

        return sol

    def get_random_init_solution(self):
        init_sol = self.parse()
        ms_seq = []
        os_seq = []
        for job_id in range(init_sol.num_jobs):
            os_seq += [job_id] * init_sol.num_process
        for idx in range(init_sol.num_jobs*init_sol.num_process):
            ms_seq.append(np.random.randint(0,init_sol.num_machine))

        np.random.seed(0)
        np.random.shuffle(ms_seq)
        np.random.shuffle(os_seq)
        init_sol.sol_seq = [ms_seq,os_seq]
        init_sol.cur_obj = init_sol.get_obj()
        # init_sol.best_obj = init_sol.cur_obj
        return init_sol

    def get_greedy_init_solution(self):   # 贪心算法求初始解
        init_sol = self.parse()
        jobs_seq = init_sol.jobs_seq
        all_jobs = []
        for idx in range(len(jobs_seq)):
            all_jobs += jobs_seq[idx]
        # 对所有工作按工序顺序以及取机器中最小的工时双排序，工时短的优先进入机器操作
        all_jobs.sort(key = lambda x : (x.process_id, min(x.work_time)))
        ms_seq = []
        os_seq = []
        for job in all_jobs:
            min_work_time = min(job.work_time)
            machine_id = job.work_time.index(min_work_time)
            job.machine_id = machine_id
            os_seq.append(job.job_id)
        for job_id in range(init_sol.num_jobs):
            for proc_id in range(init_sol.num_process):
                # print(init_sol.get_process(job_id, proc_id).job_id,
                #       init_sol.get_process(job_id, proc_id).process_id,
                #       init_sol.get_process(job_id, proc_id).machine_id)
                ms_seq.append(init_sol.get_process(job_id, proc_id).machine_id)

        init_sol.sol_seq = [ms_seq, os_seq]
        init_sol.cur_obj = init_sol.get_obj()
        return init_sol
