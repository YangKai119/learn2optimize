## 背包问题

#### 1、0-1背包问题

KpDataGen.py -- 生成相关的0-1背包问题虚拟数据。

KpSeq.py -- 设置使用到的数据结构以及相应的类方法。

KpModel.py -- 解析数据输入到相应的数据结构进行存储。

OptAlgorithm.py -- 使用暴力穷举、动态规划、回溯递归和分支界定四种精确算法进行求解。

RunKpModel.py -- 读取数据、调用模型。

#### 2、二维多重背包问题

MkpDataGen.py -- 生成相关Mkp问题虚拟数据。

MkpSeq.py -- 设置使用到的数据结构以及相应的类方法。

MkpModel.py -- 解析数据输入到相应的数据结构进行存储。

OptAlgorithm.py -- 通过添加了按单位价值排序的改进算子来设计差分进化算法进行求解。

RunMkpModel.py -- 读取数据、调用模型并画出适应度变化曲线。
