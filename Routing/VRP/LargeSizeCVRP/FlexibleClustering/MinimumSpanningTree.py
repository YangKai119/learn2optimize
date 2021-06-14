
import sys
import networkx as nx
import numpy as np
import time
import random
import numba
from numba import jit
# from functools import lru_cache

@jit(nopython=True) # jit，numba装饰器中的一种
def random_matrix_genetor(vex_num=10):
    '''
    随机图顶点矩阵生成器
    输入：顶点个数，即矩阵维数
    '''
    data_matrix=[]
    for i in range(vex_num):
        one_list=[]
        for j in range(vex_num):
            one_list.append(random.randint(1, 100))
        data_matrix.append(one_list)
    return data_matrix


def get_tree_prim(graphMatrix):
    '''
    prim 算法
    '''
    treeDis = graphMatrix[0]  # 各个点距离生成树的最短距离列表
    visited = [0 for i in range(len(graphMatrix))]  # 已经访问过的节点将被置为1
    visited[0] = 1
    # 不在树中的点距离树有最短距离，在树中对应的距离最小的那个点
    # 比如neighbor[1]=0表示在节点1还不在树中时，它离树中的节点0距离最小
    neighbor = [0] * (len(graphMatrix))
    for i in range(len(graphMatrix)):
        minDis = sys.maxsize
        # minDisPos = int()
        # 找出此时离树距离最小的不在树中顶点
        for j in range(len(graphMatrix)):
            if (not visited[j]) and (treeDis[j] < minDis):
                minDis = treeDis[j]
                minDisPos = j
    
        visited[minDisPos] = 1
        for j in range(len(graphMatrix)):
            # 刷新剩下的顶点距离树的最短距离
            if (not visited[j]) and (graphMatrix[j][minDisPos] < treeDis[j]):
                treeDis[j] = graphMatrix[j][minDisPos]
                neighbor[j] = minDisPos
    
    # print('MST:',sum(treeDis))

    return np.sum(treeDis)
 
 
 
# if __name__=='__main__':
#     time1 = time.time()
#     # graphMatrix = [[0, 54, 32, 7, 50, 60], 
#     #                 [54, 0, 21, 58, 76, 69], 
#     #                 [32, 21, 0, 35, 67, 66],
#     #                 [7, 58, 35, 0, 50, 62], 
#     #                 [50, 76, 67, 50, 0, 14], 
#     #                 [60, 69, 66, 62, 14, 0]]
    
#     graphMatrix=random_matrix_genetor(2000)
#     for i in range(len(graphMatrix)):
#         for j in range(len(graphMatrix)):
#             if i==j:
#                 graphMatrix[i][j] = 0
                
#             graphMatrix[i][j] = graphMatrix[j][i]
    
#     print('\r算法运行总时间：',time.time()-time1) 
    
#     d = get_tree_prim(graphMatrix)
    
#     print('\r算法运行总时间：',time.time()-time1)  
