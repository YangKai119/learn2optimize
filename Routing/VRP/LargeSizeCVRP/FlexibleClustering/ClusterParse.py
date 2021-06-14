
from Seq import *
import copy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

class ClusParse():
    def __init__(self,problem,clus_num):
        self.problem = problem
        self.clus_num = clus_num

    def parse(self):
        manager = ClusterManager()
        clus_cent = self.get_kmeans_clus_cent()
        clus_cent_radius = self.get_radius(clus_cent)
        for clus_idx in range(self.clus_num):
            cluster = Cluster(clus_id=clus_idx+1,        # 从1开始
                              x=clus_cent[clus_idx,0],
                              y=clus_cent[clus_idx,1],
                              radius=clus_cent_radius[clus_idx])
            manager.clusters.append(cluster)
        for idx,customer in enumerate(self.problem):
            manager.num_nodes += 1
            manager.nodes.append(CustNode(node_id=idx,
                                          node_code=customer[0],
                                          x=customer[1],
                                          y=customer[2],
                                          demand=customer[3]))
        return manager

    def get_kmeans_clus_cent(self):  # 随机生成初始质心
        clustering = KMeans(n_clusters=self.clus_num, random_state=5)
        clustering.fit(self.problem[:, 1:3])
        clus_cent = clustering.cluster_centers_
        return clus_cent

    def get_agnes_clus_cent(self):
        clustering = AgglomerativeClustering(n_clusters=self.clus_num)
        clustering.fit(self.problem[:, 1:3])
        label = clustering.labels_
        clus_cent = np.zeros((self.clus_num,2))
        for idx in range(len(label)):
            clus = label[idx]
            clus_cent[clus][0] += self.problem[idx][1]
            clus_cent[clus][1] += self.problem[idx][2]

        for clus in range(self.clus_num):
            node_num = list(label).count(clus)
            clus_cent[clus][0] = clus_cent[clus][0] / node_num
            clus_cent[clus][1] = clus_cent[clus][1] / node_num

        return clus_cent


    def get_cent_dist_mat(self,clus_cent):
        cent_dist_mat = np.zeros((self.clus_num,self.clus_num))
        for i in range(self.clus_num):
            for j in range(self.clus_num):
                if i == j:
                    cent_dist_mat[i][j] = 9999999
                else:
                    lat1,lat2 = clus_cent[i, 1],clus_cent[j, 1]
                    delta_long = np.radians(clus_cent[i, 0] - clus_cent[j, 0])  # 将角度转换为弧度
                    delta_lat = np.radians(clus_cent[i, 1] - clus_cent[j, 1])
                    dist = 2 * np.arcsin(np.sqrt(
                        pow(np.sin(delta_lat / 2), 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * pow(
                            np.sin(delta_long / 2), 2)))
                    dist = dist * 6378.2 * 1000 * 1.71

                    cent_dist_mat[i][j] = cent_dist_mat[j][i] = dist
        return cent_dist_mat

    # 计算每个聚类中心的半径，半径内根据密度进行聚类
    def get_radius(self,clus_cent):
        clus_cent_radius = []
        cent_dist_mat = self.get_cent_dist_mat(clus_cent)
        for index in range(len(clus_cent)):
            radius = cent_dist_mat[index].min() * 0.9  # 半径设置为离最近聚类中心的一半
            clus_cent_radius.append(radius)

        return clus_cent_radius