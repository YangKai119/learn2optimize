
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from MinimumSpanningTree import get_tree_prim
import sys
from ClusterParse import *
import warnings
warnings.filterwarnings('ignore')

def read_data(path):
    ls_cust = pd.read_excel(path)
    data = ls_cust[['CUST_LICENCE_CODE', 'LONGITUDE', 'LATITUDE', 'QTY_ORDER_AVG']].astype('float')
    data = np.array(data)
    return data

def get_flex_points(data,clus_num):    # 生成弹性点，弹性点数量在总数的10%左右为宜
    clus_parse = ClusParse(data,clus_num)
    manager = clus_parse.parse()
    i = 1
    for node_idx in range(manager.num_nodes):
        node = manager.get_node(node_idx)
        node_to_cent_dist = manager.cal_node_to_cents_dist(node)
        node_to_cent_dist.sort(key=lambda x:x[1])
        manager.add_cluster_node(node,node_to_cent_dist)
        # 查看进度
        sys.stdout.write('\r%s%%' % (int(i / len(data) * 100)))
        sys.stdout.flush()
        i += 1

    return manager

def flex_clustering(data, clus_num):
    manager = get_flex_points(data, clus_num)
    print('\n弹性点个数：', len(manager.flex_points))
    manager.get_init_clus_state()    # 初始化一下各自簇的距离矩阵、更新了聚类中心的坐标、初始化各簇的密度和变异系数
    # i = 1
    while manager.flex_points:
    # for flex_node in manager.flex_points:
        flex_node = manager.flex_points.pop(0)
        flex_node_to_cent_dist = manager.cal_node_to_cents_dist(flex_node)
        flex_node_to_cent_dist.sort(key=lambda x:x[1])
        manager.add_cluster_flex_node(flex_node,flex_node_to_cent_dist)
        if not flex_node.clus_id:
            manager.flex_points.append(flex_node)
        print(len(manager.flex_points))

        # # 查看进度
        # sys.stdout.write('\r%s%%' % (int(i / len(manager.flex_points) * 100)))
        # sys.stdout.flush()
        # i += 1

    return manager

def cluster_plot(data,clus_num):   # 还原成numpy或者pandas的结构进行画图
    # manager = get_flex_points(data,clus_num)
    manager = flex_clustering(data, clus_num)
    result = []
    for node_idx,customer in enumerate(manager.nodes):
        row = list(data[node_idx])
        row.append(customer.clus_id)
        result.append(row)
    result = np.array(result)
    result_df = pd.DataFrame(result, columns=[['CUST_LICENCE_CODE', 'LONGITUDE', 'LATITUDE', 'QTY_ORDER_AVG','DIST_AREA_CODE']])
    result_df.astype({'DIST_AREA_CODE':'int32'})

    result_df.to_excel('../data/result.xlsx',encoding='utf-8')
    # print(result_df)
    # print(result_df.dtypes)
    # sns.lmplot(result[:,1],result[:,2], hue=result[:,-1], data=result, fit_reg=False)
    # # sns.lmplot('LONGITUDE', 'LATITUDE', hue='DIST_AREA_CODE', data=result_df, fit_reg=False)
    # # plt.scatter(test_cent_df['LONGITUDE'], test_cent_df['LATITUDE'], c='black')
    # plt.show()

if __name__ == "__main__":
    path = '../data/ls_cust.xlsx'
    data = read_data(path)
    clus_num = 7
    # clus_parse = ClusParse(data,clus_num)
    # manager = clus_parse.parse()
    # print(manager.num_nodes)
    # manager = get_flex_points(data, clus_num)
    cluster_plot(data, clus_num)

