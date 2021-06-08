#教学任务指派问题（匈牙利法）
import numpy as np
from scipy.optimize import linear_sum_assignment

def printf(row_ind,col_ind):#输出
    print("最优教师课程指派：")
    for i in range(len(row_ind)):
        print("教师",row_ind[i],"->课程",col_ind[i],end='; ')
    print()
#教师与课程一样多
#各个教师对各个课的擅长程度矩阵
goodAt =np.array([[18,5,7,16],[10,16,6,5],[11,6,4,7],[13,12,9,11]])
weakAt=20-goodAt
row_ind,col_ind=linear_sum_assignment(weakAt)
print(row_ind)#开销矩阵对应的行索引
print(col_ind)#对应行索引的最优指派的列索引
print(goodAt[row_ind,col_ind])#提取每个行索引的最优指派列索引所在的元素，形成数组
print(goodAt[row_ind,col_ind].sum())#数组求和
printf(row_ind,col_ind)

#当教师少课程多
#各个教师对各个课的擅长程度矩阵
goodAt =np.array([[7, 3, 7, 4, 5, 5],[7, 3, 7, 4, 5, 5],
       [4, 9, 2, 6, 8, 3],[4, 9, 2, 6, 8, 3],
       [8, 3, 5, 7, 6, 4],[8, 3, 5, 7, 6, 4],
       [4, 6, 2, 3, 7, 8],[4, 6, 2, 3, 7, 8]])
weakAt=10-goodAt
print(weakAt)
row_ind,col_ind=linear_sum_assignment(weakAt)
print(row_ind)#开销矩阵对应的行索引
print(col_ind)#对应行索引的最优指派的列索引,列是教师，里面的数字代表课程（任务）
print(goodAt[row_ind,col_ind])#提取每个行索引的最优指派列索引所在的元素，形成数组
print(goodAt[row_ind,col_ind].sum())#数组求和
printf(row_ind,col_ind)


#教师少课程多且一个教师最多教两门课，最少一门
goodAt =np.array([[7,3,7,4,5,5,0,0],[7,3,7,4,5,5,100,100],
                [4, 9, 2, 6, 8, 3,0,0],[4, 9, 2, 6, 8, 3,100,100],
                [8, 3, 5, 7, 6, 4,0,0],[8, 3, 5, 7, 6, 4,100,100],
                [4, 6, 2, 3, 7, 8,0,0],[4, 6, 2, 3, 7, 8,100,100]])
weakAt=100-goodAt
row_ind,col_ind=linear_sum_assignment(weakAt)
print(row_ind)#开销矩阵对应的行索引
print(col_ind)#对应行索引的最优指派的列索引
print(goodAt[row_ind,col_ind])#提取每个行索引的最优指派列索引所在的元素，形成数组
print(goodAt[row_ind,col_ind].sum())#数组求和
printf(row_ind,col_ind)

