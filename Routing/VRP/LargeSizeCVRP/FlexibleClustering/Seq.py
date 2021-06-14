import numpy as np
import copy
from MinimumSpanningTree import get_tree_prim

class CustNode(object):
    """
    Class to represent each node for vehicle routing.
    """
    def __init__(self, node_id, node_code,x, y, demand, clus_id=None):
        self.node_id = node_id
        self.node_code = node_code
        self.x = x
        self.y = y
        self.demand = demand
        self.clus_id = clus_id

class NodeManager(object):
    """
    Base class for sequential input data. Can be used for vehicle routing.
    """
    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]

class Cluster(NodeManager):
    def __init__(self,clus_id,x,y,radius):
        super(Cluster, self).__init__()
        self.clus_id = clus_id
        self.demand = 0
        self.density = 0
        self.radius = radius   # 聚类半径
        self.cv = 0    # 变异系数
        self.x = x    # 聚类中心的坐标
        self.y = y
        self.dist_seq = []
        self.nodes_seq = []
        self.demand_seq = []
        self.clus_all_x = 0
        self.clus_all_y = 0
        self.cust_num = 0
        self.demand_std = 0
        self.demand_mean = 0
        self.dist_mat = None

    def add_node(self, node, node_to_cent_dist, update_flex_state=False):  # 维护一个更新聚类中心坐标的开关
        node.clus_id = int(self.clus_id)
        self.cust_num += 1
        self.dist_seq.append(node_to_cent_dist)
        self.demand_seq.append(node.demand)
        self.nodes_seq.append(node)
        self.demand += node.demand
        self.clus_all_x += node.x
        self.clus_all_y += node.y
        self.demand_mean = np.mean(self.demand_seq)
        self.demand_std = np.std(self.demand_seq, ddof=1)  # ddof=1为无偏样本标准差
        if update_flex_state:     # 在生成弹性点的初期不用更新聚类中心坐标以及需求均值和方差
            self.update_cent_pos()



    def remove_node(self, node, seq_idx=None, update_flex_state=False):   # 删除点
        node.clus_id = None
        self.cust_num -= 1
        if not seq_idx:
            seq_idx = self.nodes_seq.index(node)
        self.dist_seq.pop(seq_idx)
        self.demand_seq.pop(seq_idx)
        self.nodes_seq.remove(node)
        self.demand -= node.demand
        self.clus_all_x -= node.x
        self.clus_all_y -= node.y
        self.demand_mean = np.mean(self.demand_seq)
        self.demand_std = np.std(self.demand_seq, ddof=1)  # ddof=1为无偏样本标准差
        if update_flex_state:
            self.update_cent_pos()


    def update_cent_pos(self):      # 更新聚类中心坐标
        self.x = self.clus_all_x / self.cust_num
        self.y = self.clus_all_y / self.cust_num

    # 计算密度
    def cal_density(self, dist_mat, cust_num):  # 因为是临时变量，所以不必每次都更新一次
        MST = get_tree_prim(dist_mat)   # 最小生成树
        density = MST / cust_num
        return density

    # 计算变异系数
    def cal_coeff_of_var(self):    # 取各区域的平均订单量
        cv = self.demand_std / self.demand_mean
        return cv

class ClusterManager(NodeManager):
    """
    The class to maintain the state for vehicle routing.
    """
    def __init__(self):
        super(ClusterManager, self).__init__()
        self.clusters = []
        self.max_clus_num = 250
        self.max_clus_demand = 11000
        self.flex_points = []

    def clone(self):
        res = ClusterManager()
        res.nodes = copy.deepcopy(self.nodes)
        res.num_nodes = copy.deepcopy(self.num_nodes)
        res.route = copy.deepcopy(self.clusters)
        return res

    def get_cluster(self,clus_idx):
        return self.clusters[clus_idx]

    def cal_node_to_cents_dist(self,node):   # 计算该点到各个聚类中心的距离，并归类
        node_to_cents_dist = []
        for clus_idx in range(len(self.clusters)):
            one_sample_distance = self.get_dist(node,self.get_cluster(clus_idx))  # 一个样本到所有质心的距离
            node_to_cents_dist.append((clus_idx,one_sample_distance))
        return node_to_cents_dist

    def get_dist(self, node_1, node_2):
        delta_long = np.radians(node_1.x - node_2.x)   # 将角度转换为弧度
        delta_lat = np.radians(node_1.y - node_2.y)
        s = 2 * np.arcsin(np.sqrt(
            pow(np.sin(delta_lat / 2), 2) + np.cos(np.radians(node_1.y)) * np.cos(np.radians(node_2.y)) * pow(
                np.sin(delta_long / 2), 2)))
        s = s * 6378.2
        return s * 1000 * 1.71

    def get_dist_mat(self,nodes_seq):
        dist_mat = np.zeros((len(nodes_seq),len(nodes_seq)))
        for i in range(len(nodes_seq)):
            for j in range(i,len(nodes_seq)):   # 减少重复计算
                if i != j:
                    node_i = nodes_seq[i]
                    node_j = nodes_seq[j]
                    dist_mat[i][j] = dist_mat[j][i] = self.get_dist(node_i,node_j)
                else:
                    dist_mat[i][j] = 9999999
        return dist_mat

    def get_init_clus_state(self):
        for clus_idx in range(len(self.clusters)):
            cluster = self.get_cluster(clus_idx)
            cluster.update_cent_pos()
            cluster.dist_seq = [self.get_dist(cluster, cluster.nodes_seq[i]) for i in range(cluster.cust_num)]
            cluster.dist_mat = self.get_dist_mat(cluster.nodes_seq)
            cluster.density = cluster.cal_density(cluster.dist_mat, cluster.cust_num)
            cluster.cv = cluster.cal_coeff_of_var()

    # 生成一个临时用以判断的距离矩阵
    def get_tmp_dist_mat(self,node,cluster,ex_node_idx=-1):  # 增加一个点时更新矩阵
        tmp_dist_mat = copy.deepcopy(cluster.dist_mat)
        if ex_node_idx != -1:                  # 若是交换模式，则先把之前点在矩阵中的信息删除
            tmp_dist_mat = np.delete(tmp_dist_mat, ex_node_idx, 0)
            tmp_dist_mat = np.delete(tmp_dist_mat, ex_node_idx, 1)
        cust_dist_list = []
        for cur_node in cluster.nodes_seq:
            cust2cust_dist = self.get_dist(cur_node,node)
            cust_dist_list.append(cust2cust_dist)  # np.ceil向上取整
        # 转换成列np数组
        cust_dist_l = np.array(cust_dist_list)
        # 转换成行np数组
        cust_dist_list.append(9999999)  # 末尾加入9999999表示自身到自身的距离
        cust_dist_h = np.array(cust_dist_list)
        # 更新矩阵
        tmp_dist_mat = np.insert(tmp_dist_mat, len(tmp_dist_mat), values=cust_dist_l, axis=1)  # 加入到矩阵的列中
        tmp_dist_mat = np.insert(tmp_dist_mat, len(tmp_dist_mat), values=cust_dist_h, axis=0)  # 加入到矩阵的行中

        return tmp_dist_mat

    def add_cluster_node(self, node, node_to_cent_dist):   # 应该是个递归
        while not node.clus_id:
            if not node_to_cent_dist:
                node.clus_id = None
                self.flex_points.append(node)
                return
            clus_idx,dist_min_values = node_to_cent_dist.pop(0)   # 取该点到每一个聚类中心的最小的距离的值并返回所在的簇
            cluster = self.get_cluster(clus_idx)
            if cluster.radius*1.2 >= dist_min_values and cluster.cust_num < 200:
                cluster.add_node(node, dist_min_values)
            elif cluster.radius >= dist_min_values and cluster.cust_num >= 200:
                if max(cluster.dist_seq) > dist_min_values:   # 判断其是否可换
                    cluster.add_node(node, dist_min_values)
                    max_val_idx = cluster.dist_seq.index(max(cluster.dist_seq))  # 找到最大值的位置
                    ex_node = cluster.nodes_seq[max_val_idx]  # 找到最大值的点
                    cluster.remove_node(ex_node, max_val_idx)  # 从该类中移除该点，因为是加入点之后才计算的索引，所以这里可以直接通过索引删除该点
                    ex_node_to_cent_dist = self.cal_node_to_cents_dist(ex_node)
                    ex_node_to_cent_dist.sort(key=lambda x: x[1])
                    self.add_cluster_node(ex_node,ex_node_to_cent_dist)
            # elif cluster.radius * 1.2 >= dist_min_values and cluster.cust_num < 180:  # 给密度松散区域一些松弛的空间
            #     cluster.add_node(node, dist_min_values, True)


    def add_cluster_flex_node(self, flex_node, flex_node_to_cent_dist):
        while not flex_node.clus_id:
            if not flex_node_to_cent_dist:
                flex_node.clus_id = -1
                return
            clus_idx, dist_min_values = flex_node_to_cent_dist.pop(0)  # 取该点到每一个聚类中心的最小的距离的值并返回所在的簇
            cluster = self.get_cluster(clus_idx)
            # 首先看一下该区域是否已经饱和
            if cluster.cust_num < self.max_clus_num:  # 若未饱和，则尝试操作
                pre_dens = cluster.density
                pre_cv = cluster.cv
                # 先尝试加入点，并计算相关指标
                tmp_dist_mat = self.get_tmp_dist_mat(flex_node, cluster)
                cluster.add_node(flex_node, dist_min_values, True)
                tmp_dens = cluster.cal_density(tmp_dist_mat, cluster.cust_num)
                tmp_cv = cluster.cal_coeff_of_var()
                if tmp_dens <= pre_dens * 1.2 and tmp_cv <= pre_cv:  # 如果成立则加入该点，加入一个松弛系数
                    cluster.dist_seq = [self.get_dist(cluster,cluster.nodes_seq[i]) for i in range(cluster.cust_num)]  # 更新各个点距离聚类中心的距离
                    cluster.density = tmp_dens
                    cluster.cv = tmp_cv
                    cluster.dist_mat = tmp_dist_mat
                else:
                    cluster.remove_node(flex_node)

            else:                 # 还是采用剔除机制，末位淘汰，采用交换的方式，如果交换后效果变好，则进行交换
                if max(cluster.dist_seq) > dist_min_values:   # 此时聚类中心一直在变化，所以不知道会不会有很大的影响，需要更新一下clus内各个点到聚类中心的距离
                    pre_dens = cluster.density
                    pre_cv = cluster.cv
                    max_val_idx = cluster.dist_seq.index(max(cluster.dist_seq))  # 找到最大值的位置
                    ex_node = cluster.nodes_seq[max_val_idx]  # 找到最大值的点
                    cluster.remove_node(ex_node, max_val_idx, True)
                    tmp_dist_mat = self.get_tmp_dist_mat(flex_node, cluster, max_val_idx)  # 交换点之后的矩阵更新有变化
                    cluster.add_node(flex_node, dist_min_values, True)
                    tmp_dens = cluster.cal_density(tmp_dist_mat, cluster.cust_num)
                    tmp_cv = cluster.cal_coeff_of_var()
                    if tmp_dens <= pre_dens and tmp_cv <= pre_cv:  # 如果成立则加入该点，因为是以交换的方式，所以要求密度要比原来好才行，不加入松弛约束
                        cluster.dist_seq = [self.get_dist(cluster, cluster.nodes_seq[i]) for i in
                                            range(cluster.cust_num)]  # 更新各个点距离聚类中心的距离
                        cluster.density = tmp_dens
                        cluster.cv = tmp_cv
                        cluster.dist_mat = tmp_dist_mat
                        # 被踢出的点重新找下家
                        ex_node_to_cent_dist = self.cal_node_to_cents_dist(ex_node)
                        ex_node_to_cent_dist.sort(key=lambda x: x[1])
                        self.add_cluster_flex_node(ex_node, ex_node_to_cent_dist)
                    else:
                        cluster.remove_node(flex_node, True)
                        cluster.add_node(ex_node, True)







