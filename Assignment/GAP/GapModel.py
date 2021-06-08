
from GapSeq import *

class GapModel():
    def __init__(self, problem):
        self.problem = problem

    def parse(self):
        sol = AssignmentManager(self.problem['workers'],self.problem['tasks'], self.problem['max_tasks_work'])
        for i,task in enumerate(self.problem['nodes']):
            sol.tasks.append(TaskNode(task_id = i,
                                           worker_ids = task['worker_ids'],
                                           work_benfit = task['work_benfit']))
        sol.get_benfit_mat()
        return sol

    # def get_greedy_init_sol(self):
    #     init_sol = self.parse()
    #     for task_id in range(init_sol.num_tasks):
    #         max_benfit = max(init_sol.benfit_mat[:,task_id])
    #         idx = list(init_sol.benfit_mat[:,task_id]).index(max_benfit)  # 是否有同样大的，若出现同样大的则选择其中之一
    #         init_sol.sol_seq.append(idx)
    #
    #     init_sol.tot_benfit = init_sol.get_obj()
    #     return init_sol

    def get_random_init_solution(self):
        init_sol = self.parse()
        part1 = [i for i in range(init_sol.num_worker)]
        part2 = [np.random.randint(0,init_sol.num_worker) for _ in range(0,init_sol.num_tasks-init_sol.num_worker)]
        init_sol.sol_seq = part1 + part2
        np.random.seed(99)
        np.random.shuffle(init_sol.sol_seq)
        init_sol.tot_benfit = init_sol.get_obj()
        return init_sol
