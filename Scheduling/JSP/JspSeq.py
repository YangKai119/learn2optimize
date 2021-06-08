
import copy

class JobNode():            # 每道工序的点
    def __init__(self, job_id, process_id, machine_id, work_time, embedding=None):
        self.job_id = job_id
        self.machine_id = machine_id
        self.process_id = process_id
        self.work_time = work_time
        self.start_time = None
        self.finish_time = None
        self.waiting_time = None
        self.embedding = embedding

class JobManager():    # 不是一维列点的序列，而是有维度的
    def __init__(self, num_jobs):
        self.num_jobs = num_jobs
        self.jobs_seq = [[] for _ in range(self.num_jobs)]  # 工序的序列

    def get_job(self,job_id):
        return self.jobs_seq[job_id]

    def get_process(self,job_id,proc_id):
        return self.jobs_seq[job_id][proc_id]

class Machine():               # 机器应该存的是工序的序列
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.process_seq = []
        self.tot_work_time = 0  # 当前的进度

    def add_job_proc(self, proc):
        self.process_seq.append(proc)
        self.tot_work_time = proc.finish_time  # 还要加等待时间，需要判断之前的工序已完成

    def clear_state(self):     # 初始化当前状态
        self.process_seq = []
        self.tot_work_time = 0

class JobScheduleManager(JobManager):
    def __init__(self, num_machine, num_jobs ,num_process):
        super(JobScheduleManager,self).__init__(num_jobs)
        self.num_machine = num_machine
        self.num_jobs = num_jobs
        self.num_process = num_process
        self.machine_list = []
        self.sol_seq = []    # 解的序列
        self.cur_obj = 0
        self.fitness = []
        # self.best_obj = 0

    def clone(self):
        res = JobScheduleManager(self.num_machine, self.num_jobs, self.num_process)
        res.sol_seq = copy.deepcopy(self.sol_seq)
        res.machine_list = copy.deepcopy(self.machine_list)
        res.jobs_seq = copy.deepcopy(self.jobs_seq)
        res.cur_obj = copy.deepcopy(self.cur_obj)
        res.fitness = copy.deepcopy(self.fitness)
        # res.best_obj = copy.deepcopy(self.best_obj)
        return res

    def get_proc_time_state(self, cur_proc ,job_id, proc_id, machine):   # 判断该工序是否可以完成，完成的条件为，之前的工序已经完成，且其完成时间小于当前的开始工作的开始时间
        if not proc_id or not machine.process_seq:    # 在机器的开头还要在工序的开头才有这个待遇
            cur_proc.waiting_time = 0
            cur_proc.start_time = machine.tot_work_time
            cur_proc.finish_time = cur_proc.start_time + cur_proc.work_time
            return

        pre_proc = self.get_process(job_id, proc_id-1)
        if pre_proc.finish_time <= machine.tot_work_time:    # machine.tot_work_time为当前机器中最后一道工序完成的时间
            cur_proc.waiting_time = 0
            cur_proc.start_time = machine.tot_work_time
            cur_proc.finish_time = cur_proc.start_time + cur_proc.work_time
        else:
            cur_proc.waiting_time = pre_proc.finish_time - machine.tot_work_time   # 表示当前机器已经排到cur_proc了，但是因为前序工作没完成，所以产生了等待时间
            cur_proc.start_time = pre_proc.finish_time   # 前一项工作完成之后才能开始下一个工作
            cur_proc.finish_time = cur_proc.start_time + cur_proc.work_time

    def get_obj(self, seq = None):
        if not seq:
            self.seq_decode()
        else:
            self.seq_decode(seq)
        obj = 0   # 关键路径的时间，优化目标--最大完工时间最小化
        for idx in range(self.num_machine):
            obj = max(obj, self.machine_list[idx].tot_work_time)
        return obj

    def machine_reset(self):     # 更新每台机器的状态
        for i in range(self.num_machine):
            machine = self.machine_list[i]
            machine.clear_state()

    def seq_decode(self, seq = None):   # 更新各个machine类当前的状态
        if not seq:
            seq = copy.deepcopy(self.sol_seq)
        else:
            self.machine_reset()
        stack = []
        for job_id in seq:
            process_id = stack.count(job_id)
            stack.append(job_id)   # 多余的内存，后期可以优化
            job_proc = self.get_process(job_id, process_id)
            machine_id = job_proc.machine_id   # 工序只能由唯一的机器来工作
            machine = self.machine_list[machine_id]
            self.get_proc_time_state(job_proc, job_id, process_id, machine)
            machine.add_job_proc(job_proc)
