## 车间作业调度问题

#### 1、JSP问题（Job Shop Scheduling Problem）

JspDataGen.py -- 生成相关的Jsp问题虚拟数据。

JspSeq.py -- 设置使用到的数据结构以及相应的类方法。

JspModel.py -- 解析数据输入到相应的数据结构进行存储，并通过greedy算法生成初始解。

OptAlgorithm.py -- 使用模拟退火以及遗传算法分别求解。其中遗传算法设置了3种交叉算子（经典交叉、模板交叉以及pox交叉），并设置了两种变异算子（swap变异和reverse变异）。

RunJspModel.py -- 读取数据、调用模型并画出适应度变化曲线以及结果的甘特图。



#### 2、FjSP问题（Flexible Job Shop Scheduling Problem）

FjspDataGen.py -- 生成相关Fjsp问题虚拟数据。

FjspSeq.py -- 设置使用到的数据结构以及相应的类方法。

FjspModel.py -- 解析数据输入到相应的数据结构进行存储，并通过greedy算法生成初始解。

OptAlgorithm.py -- 由于Fjsp问题的复杂性，设置了MS和OS双层编码解码机制，并使用模拟退火以及遗传算法分别求解。

RunFjspModel.py -- 读取数据、调用模型并画出适应度变化以及结果的甘特图。
