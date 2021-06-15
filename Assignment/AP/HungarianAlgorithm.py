# 教学任务指派问题（匈牙利法）
from scipy.optimize import linear_sum_assignment
import numpy as np

def printf(row_ind,col_ind):# 输出
    print("最优教师课程指派:")
    for i in range(len(row_ind)):
        print("教师", row_ind[i], "->课程", col_ind[i], end='; ')
    print('\n')

# 教师与课程一样多
# 各个教师对各个课的擅长程度矩阵
ben = []
num = 10
for i in range(num):
    ben_i = []
    for j in range(num):
        ben_i.append(np.random.randint(3,18))
    ben.append(ben_i)
benfit = np.array(ben)
proc = 20 - benfit
row, col = linear_sum_assignment(proc)
print(row)      # 开销矩阵对应的行索引
print(col)      # 对应行索引的最优指派的列索引
print(benfit[row, col])    # 提取每个行索引的最优指派列索引所在的元素，形成数组
print(benfit[row, col].sum())   # 数组求和
printf(row, col)

#当教师少课程多
ben = []
teach_num = 13
work_num = 20
for i in range(teach_num):
    ben_i = []
    for j in range(work_num):
        ben_i.append(np.random.randint(3,18))
    ben.append(ben_i * 2)   # 扩展矩阵
benfit = np.array(ben)
proc = 20 - benfit
row, col=linear_sum_assignment(proc)
printf(row, col)

# 教师少课程多且一个教师最多教两门课，最少一门
benfit = np.array([[7, 3, 7, 4, 5, 5, 0, 0],[7, 3, 7, 4, 5, 5, 100, 100],
                    [4, 9, 2, 6, 8, 3, 0, 0],[4, 9, 2, 6, 8, 3, 100, 100],
                    [8, 3, 5, 7, 6, 4, 0, 0],[8, 3, 5, 7, 6, 4, 100, 100],
                    [4, 6, 2, 3, 7, 8, 0, 0],[4, 6, 2, 3, 7, 8, 100, 100]])
proc = 100 - benfit
row, col = linear_sum_assignment(proc)
printf(row, col)

