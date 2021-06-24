import numpy as np
import copy

class BpNode(object):
    def __init__(self, node_id,length, width, num, embedding=None):
        self.node_id = node_id
        self.bin_id = -1
        self.length = length
        self.width = width
        self.area = length*width
        self.num = num
        self.direction = 0
        self.set_point = None
        self.embedding = embedding
        # self.rest_space = None
        # self.new_space = None

class SeqManager(object):
    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]

class Bin():
    def __init__(self, bin_id):
        self.bin_id = bin_id
        self.goods_seq = []
        self.tot_wgt = 0

class BpManager(SeqManager):

    def __init__(self, bin_length, bin_width):
        super(BpManager, self).__init__()
        self.bin_length = bin_length    # x
        self.bin_width = bin_width     # y
        self.bin_areas = bin_width*bin_length
        self.bins = []
        self.direction_seq = []   # 装箱方向编码，3维的为0，1，2，3，4，5 一共6个方向
        self.order_seq = []       # 装箱顺序编码
        self.sol_seq = []
        self.tot_goods_areas = 0   # 货物的总体积
        self.features = []
        self.best_fit = 0   # 箱子的个数，越少越好 or 空间利用率最大，即每个箱子--当前物品体积之和/总容积 --> 其实就是所有物品的体积之和/箱子的总容积
        self.init_fit = 0
        self.bin_num = 0    # 记录箱子数量
        self.rest_space = []    # 存入空间编码，前
        self.fitness = []

    def clone(self):
        res = BpManager(self.bin_length, self.bin_width)
        res.nodes = copy.deepcopy(self.nodes)
        res.init_fit = self.init_fit
        res.tot_goods_areas = self.tot_goods_areas
        res.num_nodes = copy.deepcopy(self.num_nodes)
        res.features = copy.deepcopy(self.features)
        res.fitness = copy.deepcopy(self.fitness)
        return res

    def decode_direction(self, goods, dic):
        if dic == 1:
            x, y = goods.width, goods.length
        else:
            x, y = goods.length,  goods.width    # 最原始的状态，x长，y宽
        return x, y
    # FF方法，找到第一个合适的就放置
    def first_fit(self, idx, dic):     # 写一个装箱的方法，然后计算剩余空间
        if not self.rest_space:
            init_space = (self.bin_num, 0, 0, self.bin_length, self.bin_width)
            init_bin = Bin(self.bin_num)
            self.bins.append(init_bin)
            self.rest_space.append(init_space)
        # print(self.rest_space)
        goods = self.get_node(idx)
        goods.bin_id = -1
        goods.direction = dic
        x, y = self.decode_direction(goods, dic)
        for i in range(len(self.rest_space)):
            space = self.rest_space[i]     # 编码为（i,a,b,L,W） --> abc为可放置点，LWH为剩余空间
            tmp_bin = self.bins[space[0]]
            # tmpx = x + space[1]
            # tmpy = y + space[2]
            # tmpz = z + space[3]
            if x <= space[3] and y <= space[4]:    # 满足装载条件后将物品加入箱子并更新剩余空间
                goods.bin_id = space[0]   # 第一个就是集装箱序号
                goods.set_point = (space[1],space[2])   # 可放置点
                tmp_bin.goods_seq.append(goods)
                new_space = self.get_new_space(space, x, y)
                # goods.new_space = new_space
                for no in range(2):
                    if new_space[no][3] != 0 and new_space[no][4] != 0:
                        self.rest_space.append(new_space[no])
                self.rest_space.remove(space)
                self.space_merge(space[0])
                # goods.rest_space = self.rest_space
                break

        if goods.bin_id == -1:         # 遍历了一圈之后发现还是没有合适的位置放置，则加入进新的箱子中
            self.bin_num += 1
            # self.rest_space = []  # 先清空状态，重新加箱子了
            new_init_space = (self.bin_num, 0, 0, self.bin_length, self.bin_width)
            new_init_bin = Bin(self.bin_num)
            self.bins.append(new_init_bin)
            self.rest_space.append(new_init_space)
            self.first_fit(idx, dic)     # 递归

    def get_new_space(self, space, a, b):   # 生成新的剩余空间，如果L,W,H比a,b,c小的话就到不了这一步
        space1 = (space[0], space[1]+a, space[2], space[3]-a, space[4])
        space2 = (space[0], space[1], space[2]+b, a, space[4]-b)
        # print(space1)
        return [space1, space2]

    def space_merge(self, bin_id):     # 空间合并
        self.rest_space.sort(key=lambda x: (x[1],x[2]))   # 从下到上，从左往右的放置规则
        # 出现剩余空间为0的直接剔除
        # print(self.rest_space)

    def get_obj(self, seq=None):     # 计算适应度值
        if not seq:
            order_seq = self.order_seq
            direction_seq = self.direction_seq
        else:
            direction_seq = seq[0]
            order_seq = seq[1]
        self.rest_space = []  # 先清空状态
        self.bins = []
        self.bin_num = 0
        for idx in range(self.num_nodes):
            node_id = order_seq[idx]
            dic = direction_seq[idx]
            self.first_fit(node_id, dic)
        # 计算空间利用率
        obj = self.tot_goods_areas/(len(self.bins)*self.bin_areas)
        return obj

