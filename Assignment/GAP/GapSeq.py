
import numpy as np
import copy

class TaskNode():
    def __init__(self, task_id, worker_ids, work_benfit, embedding=None):
        self.task_id = task_id
        self.worker_ids = worker_ids
        self.work_benfit = work_benfit
        self.embedding = embedding

class TaskManager():
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.tasks = []

    def get_task(self, idx):
        return self.tasks[idx]

class AssignmentManager(TaskManager):
    def __init__(self, num_worker, num_tasks, max_tasks_work):
        super(AssignmentManager, self).__init__(num_tasks)
        self.num_worker = num_worker
        self.num_tasks = num_tasks
        self.max_tasks_work = max_tasks_work  # 每个人最多领取的工作数
        self.sol_seq = []
        self.tot_benfit = 0
        self.benfit_mat = []
        self.fitness = []

    def clone(self):
        res = AssignmentManager(self.num_worker, self.num_tasks)
        res.sol_seq = copy.deepcopy(self.sol_seq)
        res.tot_benfit = copy.deepcopy(self.tot_benfit)
        res.benfit_mat = copy.deepcopy(self.benfit_mat)
        res.fitness = copy.deepcopy(self.fitness)
        return res

    def get_benfit_mat(self):
        for task in self.tasks:
            self.benfit_mat.append(task.work_benfit)
        self.benfit_mat = np.array(self.benfit_mat).T  # 变成工人*任务的形式


    def seq_decode(self, seq=None):
        if not seq:
            seq = self.sol_seq[:]
        ass_mat = np.zeros((self.num_tasks,self.num_worker))
        for task_id in range(self.num_tasks):
            worker_id = seq[task_id]
            ass_mat[task_id][worker_id] = 1
        return np.array(ass_mat)

    def get_obj(self, seq=None):
        if not seq:
            ass_mat = self.seq_decode()
        else:
            ass_mat = self.seq_decode(seq)
        obj = 0
        for task_id in range(self.num_tasks):
            obj += np.dot(self.benfit_mat[:,task_id],ass_mat[task_id])
        return obj

