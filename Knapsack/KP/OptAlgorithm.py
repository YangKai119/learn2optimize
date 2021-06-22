"""
四种精确求解算法
1.暴力穷举
2.动态规划
3.回溯
4.分支界定
"""

import copy
import numpy as np

class BruteForce():                # 使用递归调用
    def __init__(self, sol):
        self.val_max = 0       # 全局最优，记录最大价值
        self.sol = sol
        self.visit_max = np.zeros(sol.num_nodes)  # 全局最优，记录最大价值时的序列
        self.vals = sol.values[:]
        self.wgts = sol.weights[:]

    def solver(self, visit=[], cur_val=0, cur_wgt=0):
        if len(visit) == 0:
            visit = np.zeros(self.sol.num_nodes)
        if cur_val > self.val_max:
            self.val_max = cur_val
            self.visit_max = copy.deepcopy(visit)
        for i in range(self.sol.num_nodes):
            if visit[i] == 0:
                if self.wgts[i] > self.sol.capacity - cur_wgt:
                    continue
                visit[i] = 1
                self.solver(visit, cur_val+self.vals[i], cur_wgt+self.wgts[i])
                visit[i] = 0
        self.sol.tot_obj = self.val_max
        self.sol.get_seq(self.visit_max)  # 获取物品序列
        # self.sol.fitness.append(self.sol.get_obj())

class DynamicProgramming():   # 动态规划
    def __init__(self, sol):
        self.sol = sol
        self.vals = sol.values[:]
        self.wgts = sol.weights[:]
        # self.vals = [1,2,3,7,8,9,6,5,4,0]
        # self.wgts = [6,5,4,1,2,3,9,8,7,0]
        self.num_nodes = sol.num_nodes
        self.capa = sol.capacity
        self.val_tab = np.zeros((self.num_nodes+1, self.capa+1))   # 价值动态规划状态表，行为第i个物体，列为背包容量

    def load_seq(self):
        seq = []
        col = self.capa
        for i in range(self.val_tab.shape[0]-1,0,-1):
            if self.val_tab[i,col] == self.val_tab[i-1,col]:
                continue
            else:
                seq.append(i)
                col -= self.wgts[i-1]
        seq.reverse()
        return seq

    def solver(self):
        for i in range(1, self.val_tab.shape[0]):
            for j in range(1, self.val_tab.shape[1]):
                if self.wgts[i-1] > j:    # 物体重量大于包当前重量则无法装入
                    self.val_tab[i,j] = self.val_tab[i-1,j]
                else:
                    update = self.val_tab[i-1,j-self.wgts[i-1]] + self.vals[i-1] # 计算更新背包后的价值是否比之前的要大
                    if self.val_tab[i-1,j] > update:
                        self.val_tab[i,j] = self.val_tab[i-1,j]
                    else:
                        self.val_tab[i,j] = update
        print(self.val_tab)
        self.sol.tot_obj = self.val_tab[-1,-1]
        self.sol.goods_seq = self.load_seq()


class BackTracking():           # 回溯法
    def __init__(self, sol):
        self.sol = sol
        self.vals = sol.values[:]
        self.wgts = sol.weights[:]
        self.capa = sol.capacity
        # self.vals = [1,2,3,7,8,9,6,5,4,20]
        # self.wgts = [6,5,4,1,2,3,9,8,7,20]
        self.cur_val = 0      # 当前价值
        self.cur_wgt = 0
        self.val_max = 0  # 全局最优，记录最大价值
        self.visit_max = np.zeros(sol.num_nodes)
        self.org_idx = []

    def sort_per_val(self):               # 求单位质量大小
        per_idx = [(i, self.vals[i]/self.wgts[i]) for i in range(self.sol.num_nodes)]
        per_idx.sort(key=lambda x: x[1], reverse=True)
        tmp_vals = self.vals[:]
        tmp_wgts = self.wgts[:]
        for i in range(self.sol.num_nodes):
            idx = per_idx[i][0]
            self.org_idx.append(idx)
            tmp_vals[i] = self.vals[idx]
            tmp_wgts[i] = self.wgts[idx]

        self.vals = tmp_vals[:]
        self.wgts = tmp_wgts[:]

    def bound(self, i):   # 定义上界函数
        delta_capa = self.capa - self.cur_wgt   # 剩余容量
        best_bound = self.cur_val
        while i < self.sol.num_nodes:
            if self.wgts[i] <= delta_capa:
                best_bound += self.vals[i]
                delta_capa -= self.wgts[i]
                i += 1
            else:
                best_bound += self.vals[i] / self.wgts[i] * delta_capa
                break
        return best_bound

    def solver(self, i=0, visit=[]):
        if len(visit) == 0:
            visit = np.zeros(self.sol.num_nodes)
            self.sort_per_val()   # 进行单位价值排序

        if i > self.sol.num_nodes - 1:
            self.val_max = self.cur_val
            self.sol.tot_obj = self.val_max
            v_seq = []
            for idx in range(self.sol.num_nodes):
                if visit[idx] != 0:
                    v_seq.append(self.org_idx[idx])
            self.sol.goods_seq = sorted(v_seq[:])  # 获取物品序列
            return

        if self.cur_wgt + self.wgts[i] < self.capa:
            self.cur_wgt += self.wgts[i]
            self.cur_val += self.vals[i]
            visit[i] = 1
            self.solver(i+1, visit)
            self.cur_val -= self.vals[i]
            self.cur_wgt -= self.wgts[i]

        if self.bound(i+1) >= self.val_max:
            self.solver(i+1, visit)

# 分支界定法
class TreeNode():   # 定义二叉树结点，来存储当前解的状态
    def __init__(self, v, w, idx, father, flag, goods_idx=-1):
        self.currv = v        # 记录之前解的节点状态
        self.currw = w
        self.idx = idx   # 节点的idx和物品的idx不一样
        self.goods_idx = goods_idx  # 表示此时加入的是哪件物品
        self.up = 0    # 用来更新上界
        self.down = 0  # 用来更新下界
        self.flag = flag   # 判断选或不选以及停止条件，-1时为虚拟节点（无物理意义）
        self.father = father
        # self.left_child = None    # 结果应该都在左边的节点中显示
        # self.right_child = None

class BranchAndBound():
    def __init__(self, sol):
        self.sol = sol
        self.vals = sol.values[:]
        self.wgts = sol.weights[:]
        self.capa = sol.capacity
        # self.vals = [1,2,3,7,8,9,6,5,4,20]
        # self.wgts = [6,5,4,1,2,3,9,8,7,20]
        self.val_max = 0  # 全局最优，记录最大价值
        self.bag = set()
        self.org_idx = []

    def sort_per_val(self):               # 求单位质量大小
        per_idx = [(i, self.vals[i]/self.wgts[i]) for i in range(self.sol.num_nodes)]
        per_idx.sort(key=lambda x: x[1], reverse=True)
        tmp_vals = self.vals[:]
        tmp_wgts = self.wgts[:]
        for i in range(self.sol.num_nodes):
            idx = per_idx[i][0]
            self.org_idx.append(idx)
            tmp_vals[i] = self.vals[idx]
            tmp_wgts[i] = self.wgts[idx]

        self.vals = tmp_vals[:]
        self.wgts = tmp_wgts[:]

    def bound(self, i, node):
        delta_w = self.capa - node.currw
        bestbound = node.currv
        while i < self.sol.num_nodes:
            if self.wgts[i] < delta_w:
                bestbound += self.vals[i]
                delta_w -= self.wgts[i]
                i += 1
            else:
                bestbound += self.vals[i] / self.wgts[i] * delta_w
                break
        return bestbound

    def solver(self):
        queue = []
        queue.append(TreeNode(0, 0, 0, None, -1))
        self.sort_per_val()  # 进行单位价值排序
        while queue:
            node = queue.pop(0)
            if node.idx < self.sol.num_nodes:
                # 解的左边表示接受将该物品放进背包
                # if not node.left_child:
                leftv = node.currv + self.vals[node.idx]
                leftw = node.currw + self.wgts[node.idx]
                left_node = TreeNode(leftv, leftw, node.idx + 1, node, 1, node.idx)
                left_node.up = self.bound(left_node.idx, left_node)
                # node.left_child = left_node
                if left_node.currw < self.capa:
                    queue.append(left_node)
                    if left_node.currv > self.val_max:   # 更新一次最优解
                        self.val_max = left_node.currv
                        # 返回序列
                        cur_node = left_node
                        # self.best_idx.clear()  # 更新当前最优解idx
                        while cur_node.flag != -1:   # 根节点为-1
                            if cur_node.goods_idx != -1:
                                self.bag.add(self.org_idx[cur_node.goods_idx])  # 返回当前点而不是left点的idx
                            cur_node = cur_node.father

                # 解的右边表示未将物品放进背包
                # if not node.right_child:
                right_node = TreeNode(node.currv, node.currw, node.idx + 1, node, 0)
                right_node.up = self.bound(right_node.idx, right_node)
                # node.right_child = right_node
                if right_node.up >= self.val_max:
                    queue.append(right_node)

        self.sol.tot_obj = self.val_max
        self.sol.goods_seq = self.bag




