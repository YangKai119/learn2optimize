
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import  LabelEncoder
import math
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class Cluster():
    def __init__(self,cust,cust_num_max=2000,cust_num_min=700,clus_num=None):
        #参数初始化
        self.cust = cust    #保持原索引不变
        self.clus_num = clus_num
        self.cust_num_max = cust_num_max
        self.cust_num_min = cust_num_min
        
    def get_keys(self,d,value):
        
        return [k for k, v in d.items() if v == value]
    
    #计算两点间的直线距离
    def get_gps_distance(self,point1,point2):
        long1, lat1 = point1
        long2, lat2 = point2
        delta_long = math.radians(long2 - long1)
        delta_lat = math.radians(lat2 - lat1)
        s = 2 * math.asin(math.sqrt(
            math.pow(math.sin(delta_lat / 2), 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.pow(
                math.sin(delta_long / 2), 2)))
        s = s * 6378.2
        return s*1000

    def clus_AGNES(self):
        #训练聚类模型
        clustering = AgglomerativeClustering(linkage='complete',n_clusters=3)
        #clustering = KMeans(n_clusters=self.clus_num,random_state=0)
        clustering.fit(self.cust[['LONGITUDE','LATITUDE']])
        self.cust["DIST_AREA_CODE"] = clustering.labels_
        
        return self.cust
    
    def clus_KMeans(self,data):
        #拆分户数大于上限的类
        while data["DIST_AREA_CODE"].value_counts().max() > self.cust_num_max:
            clusters = data["DIST_AREA_CODE"].drop_duplicates().tolist()
            for clu in clusters:
                clus_0 = data[data["DIST_AREA_CODE"]==clu]
                if len(clus_0) > self.cust_num_max:
                    clu_num = int(np.ceil(len(clus_0)/self.cust_num_max))
                    clustering0 = KMeans(n_clusters=clu_num, random_state=0)
                    clustering0.fit(clus_0[['LONGITUDE','LATITUDE']].values)
                    clus_0["DIST_AREA_CODE"] = clustering0.labels_+data["DIST_AREA_CODE"].max()+1
                    #print(pd.value_counts(clus_0["DIST_AREA_CODE"]))
                    #print(clustering0.cluster_centers_)
                    
                    clus_0_clu=clus_0[["CUST_LICENCE_CODE","DIST_AREA_CODE"]]
                    data=data.merge(clus_0_clu,on="CUST_LICENCE_CODE",how="left",suffixes=["",'_new'],copy = False)
                    new_clu_index=data[data["DIST_AREA_CODE_new"].notnull()].index
                    data.loc[new_clu_index,"DIST_AREA_CODE"]=data.loc[new_clu_index,"DIST_AREA_CODE_new"]     #新的索引替代旧的索引
                    data = data.drop("DIST_AREA_CODE_new",axis=1) 
            data['DIST_AREA_CODE']=LabelEncoder().fit_transform(data['DIST_AREA_CODE'])
        
        return data
    
    def clus_merge(self,data):
        clus_cent = data[['LONGITUDE','LATITUDE']].groupby(data['DIST_AREA_CODE']).mean().reset_index()
        #生成聚类中心的字典
        clus_cent_dict = dict(zip([x for x in clus_cent['DIST_AREA_CODE']],[tuple(x) for x in clus_cent[['LONGITUDE','LATITUDE']].values]))       
        clus_custnum = data["DIST_AREA_CODE"].value_counts().to_dict()
        #给类别数排序，先从户数最少的类开始合并
        clus_custnum_sort = sorted(clus_custnum.items(),key=lambda x:x[1],reverse=False)  
        #约束每类户数不低于cust_min户
        while data["DIST_AREA_CODE"].value_counts().min() < self.cust_num_min:
            
            for clu in clus_custnum_sort:
                clu = clu[0]
                clus_0 = data[data["DIST_AREA_CODE"]==clu]
                if len(clus_0) < self.cust_num_min:
                    #（层次聚类思想）通过计算户数低于cust_min户的类到其他类的距离，选择最近的类进行合并
                    #建立一个新的DataFrame存放距离和区域
                    clus_cent_dist = pd.DataFrame(columns = ['dist','area_from','area_to'])
                    
                    for i in clus_cent_dict.keys():
                        if i != clu:
                            clus_cent_dist_j = self.get_gps_distance(clus_cent_dict[clu],clus_cent_dict[i]) #一个样本到所有质心的距离 
                            clus_cent_dist = clus_cent_dist.append(pd.DataFrame({'dist':[clus_cent_dist_j],'area_from':[clu],'area_to':[i]}),ignore_index=True)
                        #print(clu)
                    #data_change = data.loc[data["DIST_AREA_CODE"] != clu]
                    #for i in data_change['CUST_LICENCE_CODE']:
                    #    clus_cent_dist_j = 1.71 * get_gps_distance(clus_cent.iloc[clu].values,data_change[['LONGITUDE','LATITUDE']].loc[data_change['CUST_LICENCE_CODE'] == str(i)].values[0]) #一个质心到所有其他类所有点的距离 
                    #    clus_cent_dist.append(clus_cent_dist_j)
                    dist_min_values = clus_cent_dist['dist'].min() #取该点到每一个聚类中心的最小的距离的值
                    dist_min_index  = clus_cent_dist['area_to'].loc[clus_cent_dist['dist'] == dist_min_values]#距离最短的的索引，where函数返回的是索引    
                    cust_index = dist_min_index.values[0]
                    data["DIST_AREA_CODE1"] = data["DIST_AREA_CODE"]
                    data["DIST_AREA_CODE1"].replace(clu,cust_index,inplace = True)
                    #若合并时户数超过cust_max户则放弃该两类合并，并寻找次近距离的类尝试合并，直到合并后的类别户数小于cust_max时循环停止
                    while (data["DIST_AREA_CODE1"] == cust_index).sum() > self.cust_num_max:
                        clus_cent_dist = clus_cent_dist.drop(clus_cent_dist.loc[clus_cent_dist['area_to']==cust_index].index)
                        dist_min_values = clus_cent_dist['dist'].min() #取该点到每一个聚类中心的最小的距离的值
                        dist_min_index  = clus_cent_dist['area_to'].loc[clus_cent_dist['dist'] == dist_min_values]#距离最短的的索引，where函数返回的是索引    
                        if len(clus_cent_dist) == 0 :
                            break               
                        cust_index = dist_min_index.values[0]
                        data["DIST_AREA_CODE1"] = data["DIST_AREA_CODE"]
                        data["DIST_AREA_CODE1"].replace(clu,cust_index,inplace = True)
                    
                    data["DIST_AREA_CODE"] = data["DIST_AREA_CODE1"]
                    data = data.drop("DIST_AREA_CODE1",axis=1)
                    
            clus_cent = data[['LONGITUDE','LATITUDE']].groupby(data['DIST_AREA_CODE']).mean().reset_index()
            clus_cent_dict = dict(zip([x for x in clus_cent['DIST_AREA_CODE']],[tuple(x) for x in clus_cent[['LONGITUDE','LATITUDE']].values]))
            clus_custnum = data["DIST_AREA_CODE"].value_counts().to_dict()
            clus_custnum_sort = sorted(clus_custnum.items(),key=lambda x:x[1],reverse=False) 
                
            data['DIST_AREA_CODE']=LabelEncoder().fit_transform(data['DIST_AREA_CODE'])
            
        return data
 
    
    def fit(self):
        if not self.clus_num:
            data = self.clus_AGNES()
            data = self.clus_KMeans(data)
            data = self.clus_merge(data)
        
        return data
    
    
    
