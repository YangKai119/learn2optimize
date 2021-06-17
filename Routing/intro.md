## 车辆路径问题

#### 1、TSP问题（Travelling Salesman Problem）

TspDataGen.py -- 生成相关的Tsp问题虚拟数据。

TspSeq.py -- 设置使用到的数据结构以及相应的类方法。

TspModel.py -- 解析数据输入到相应的数据结构进行存储，并通过greedy算法生成初始解。

OptAlgorithm.py -- 使用模拟退火以及禁忌搜索算法分别求解。禁忌搜索算法依赖领域搜索能力较强的算子。

RunTspModel.py -- 读取数据、调用模型并画出适应度变化曲线以及车辆路径图。



#### 2、VRP问题（Vehicle Routing Problem）

ACO_CVRP.py -- 用蚁群算法求解CVRP问题（使用solomon100数据集）。

ORtools_CVRP.py -- 使用ORtools工具求解CVRP问题（使用solomon100数据集）。

VrpParse.py -- 将数据处理成ORtools所能接受的数据输入形式。

LargeSizeCVRP -- 求解大规模CVRP问题的方法


#### 3、Learing4Routing（利用深度强化学习求解路径问题）

TspUntils.py -- 生成Tsp实例数据以及需要使用的运算操作方法。

Network.py -- 建立mask、Attention机制以及pointer network网络。

RunTspModel.py -- 引入相关模型训练并求解。
