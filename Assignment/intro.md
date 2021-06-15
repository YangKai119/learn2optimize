## 指派问题

#### 1、经典指派问题（Assignment Problem）

Gurobi_AP.ipynb -- 建立一个20*20的打车订单分配问题，顾客与司机一一对应进行服务，并调用Gurobi求解器进行求解。

HungarianAlgorithm.py -- 调用scipy库中的匈牙利算法来求解教学任务指派问题。



#### 2、广义指派问题（Generalized Assignment Problem）

GapDataGen.py -- 生成相关的Gap问题虚拟数据。

GapSeq.py -- 设置使用到的数据结构以及相应的类方法。

GapModel.py -- 解析数据输入到相应的数据结构进行存储，并通过greedy算法生成初始解。

OptAlgorithm.py -- 使用改进粒子群算法进行求解。并设置了一个适用于Gap问题的按位解码方式。

RunGapModel.py -- 读取数据、调用模型并画出适应度变化曲线。





