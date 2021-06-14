# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:32:04 2021

@author: omgya
"""


import math
import numpy as np
import pandas as pd
import warnings
import sys
import time
from CustClustering import Cluster
from ORTools import OR_Tools

warnings.filterwarnings('ignore')

#数据处理
class GetData():
    
    def __init__(self):
        pass
    
    def Customer(self):
        ls_cust = pd.read_excel(r'D:\办公文件\研究生项目\路径优化研究\代码\毕设代码\启发式算法求解VRP\data\LS_CUST.xlsx')
        #ls_order先取一周（一个订货周期）的数据做测试
        ls_order = pd.read_excel(r'D:\办公文件\研究生项目\路径优化研究\代码\毕设代码\启发式算法求解VRP\data\LS_ORDER_WEEKS.xlsx')
        data_ord = ls_order[['CUST_LICENCE_CODE','QTY_ORDER_SUM']]
        data_ord['CUST_LICENCE_CODE'] = data_ord['CUST_LICENCE_CODE'].astype('str')
        data_ord_sum = data_ord.groupby(['CUST_LICENCE_CODE'],as_index=False).mean()
        data_ord_sum['QTY_ORDER_AVG'] = data_ord_sum['QTY_ORDER_SUM'].round(2)
        data_ord_avg = data_ord_sum.drop(columns = 'QTY_ORDER_SUM')
        data_lonlat = ls_cust[['CUST_LICENCE_CODE','LONGITUDE','LATITUDE']]
        data_lonlat['CUST_LICENCE_CODE'] = data_lonlat['CUST_LICENCE_CODE'].astype('str')
        cust_data = pd.merge(data_ord_avg,data_lonlat,on = 'CUST_LICENCE_CODE')
        cust_data['SERVICE_TIME'] = 1.5    #后期需要随机生成数据
        ls_delivery_location = pd.DataFrame([['135001003200',0,119.197559,26.069828,0]],columns = cust_data.columns)
        #将发货点信息加入到零售户数据表中
        cust_df = ls_delivery_location.append(cust_data).reset_index(drop = True)
        
        return cust_df
    
    def Vehicle(self):
        ls_vehicle = pd.read_excel(r'D:\办公文件\研究生项目\路径优化研究\代码\毕设代码\启发式算法求解VRP\data\LS_VEHICLE.xlsx')
        ls_vehicle = ls_vehicle.loc[ls_vehicle['THIRD_PARTY']==0].reset_index(drop=True)
        vehicles_df = ls_vehicle[['DIST_STATION_CODE','VEHICLE_CODE','MAX_LOAD_PACKAGE','MAX_LOAD_CUST']]  #车辆数据取发货点、车牌、最大载重量、最大服务客户数
        vehicles_df['DRIVE_TIME'] = 0
        vehicles_df['WORK_COUNTS'] = 0
        # vehicles_df['TOTAL_ROUTE_PLAN'] = None
        vehicles_df['DIST_STATION_CODE'] = vehicles_df['DIST_STATION_CODE'].astype('str')
        # vehicles = dict(zip(vehicles_df.index, vehicles_df[['DIST_STATION_CODE','VEHICLE_CODE','MAX_LOAD_PACKAGE','MAX_LOAD_CUST']].values))
        vehicles_df = vehicles_df.sort_values('MAX_LOAD_PACKAGE').reset_index(drop=True)
        
        return vehicles_df
   
class Processing():
    def __init__(self):
        pass
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
    
    def cal_cust_dist(self,x,y,dl_c_gps):
        if x == y:
            return 0
        else:
            return int(self.get_gps_distance(dl_c_gps[x], dl_c_gps[y]) * 1.71)   #1.71为实际距离与直线距离之间的系数
    
    #计算驾驶时间
    def cal_cust_drv_time(self,x,y,dl_c_gps):
        if x == y:
            return 0
        else:
            return int(self.get_gps_distance(dl_c_gps[x],dl_c_gps[y])*1.8 /1000/60*3600)
    
    def get_matrix(self,cust_df):
        
        #发货点
        dl_c_gps = dict(zip(cust_df["CUST_LICENCE_CODE"], cust_df[["LONGITUDE", "LATITUDE"]].values))
        cust_run = cust_df["CUST_LICENCE_CODE"].tolist()
        print("Matrix construction:")
        
        i = 1
        time1 = time.time()
        distance_matrix = []
        for cust in cust_run:
            distance_matrix.append([self.cal_cust_dist(cust,y,dl_c_gps) for y in cust_run])
            sys.stdout.write('\r%s%%'%(int(i/len(cust_run)*100)))
            sys.stdout.flush()
            i += 1
            
        distance_matrix = np.array(distance_matrix)    #需要转换成np.array
        
        print("\rTotal run time:{}".format(time.time()-time1))    
        print(distance_matrix)
        
        return distance_matrix    


# VRP求解
class VRP():
    
    def __init__(self,cust_df,vehicles_df,dist_mat=None):
        
        self.cust_df = cust_df
        self.cust = cust_df.drop(index=0)
        self.vehicles_df = vehicles_df
        self.vehicles = dict(zip(vehicles_df.index,vehicles_df[['DIST_STATION_CODE','VEHICLE_CODE','MAX_LOAD_PACKAGE','MAX_LOAD_CUST']].values))
        self.dist_mat = dist_mat
        
    def get_keys(self,d,value):
        
        return [k for k, v in d.items() if v == value]
    
    def get_cluster(self):
        #获取聚类结果
        cust = Cluster(self.cust).fit()
        
        return cust
    
    def get_matrix(self,cust_df):
        #获取距离矩阵
        pro = Processing()
        distance_matrix = pro.get_matrix(cust_df)
        
        return distance_matrix    
    
    #单进程
    def single_process(self,dist_mat_separate_gen=0):

        cust = self.get_cluster()
        print('聚类结束')
        cust['DELIVER_NO'] = None
        cust['DELIVER_NAME'] = None
        cust_data = pd.DataFrame(columns = cust.columns)     #生成配送顺序后进行输入
        clus_list = cust['DIST_AREA_CODE'].drop_duplicates().tolist()
        num = 0  #获取路线编码
        dist_mat_separate_gen = 1  #是否分开生成距离矩阵，1为是，0为否
        #每个区域分别跑OR-tools
        car_plan = pd.DataFrame(columns=self.vehicles_df.columns)
        vehicles_df = self.vehicles_df.copy()
        for clus in clus_list:
            self.vehicles_df['CLUS_{}_ROUTE_PLAN'.format(clus)]=0
            self.vehicles_df['CLUS_{}_ROUTE_PLAN'.format(clus)]=self.vehicles_df['CLUS_{}_ROUTE_PLAN'.format(clus)].apply(lambda x:[0,0])
            
            cust_0 = cust.loc[cust['DIST_AREA_CODE']==clus]
            cust_0 = cust_0[['CUST_LICENCE_CODE','LONGITUDE','LATITUDE','QTY_ORDER_AVG','SERVICE_TIME','DIST_AREA_CODE','DELIVER_NO','DELIVER_NAME']]
            #发货点
            ls_delivery_location = pd.DataFrame([['135001003200',119.197559,26.069828,0,0,'',0,'发货点']],columns = cust_0.columns)
            #将发货点信息加入到零售户数据表中
            cust_0 = ls_delivery_location.append(cust_0)
            
            #是否分开生成矩阵
            if not dist_mat_separate_gen:            
                cust_0_ids_all = cust_0['CUST_LICENCE_CODE']    # 不分开生成矩阵
                clus_0_ids = cust_0["CUST_LICENCE_CODE"].apply(lambda x:self.get_keys(cust_0_ids_all,x)[0]).tolist()
                #dist_mat=np.array(dist_mat)
                dist_cust_mat_0 = self.dist_mat[clus_0_ids,:][:,clus_0_ids]   #直接在大矩阵中取数
            else:
                dist_cust_mat_0 = self.get_matrix(cust_0)      # 分开生成矩阵
            
            
            drv_time_mat = np.asarray(dist_cust_mat_0) * 6 / 100    #生成驾驶时间矩阵
            #得到每辆车的送货线路(在此处更新车辆工作时间、往返趟次信息)

            route_plan_0,vehicles_df = OR_Tools(cust_0,vehicles_df,dist_cust_mat_0,drv_time_mat).main()
            car_plan = 'CLUS_{}_ROUTE_PLAN'.format(clus)
            vehicles_df[car_plan] = route_plan_0
            # #统计路线数
            for i in vehicles_df.index:
                car_clus_route = vehicles_df[car_plan].loc[i]
                if len(car_clus_route) > 2:
                    vehicles_df['WORK_COUNTS'].loc[i] += 1
                    
            self.vehicles_df.loc[vehicles_df.index]=vehicles_df.values
            
            #更新车辆信息看看是否能在ORtools里面尝试做限制
            #在这个位置加入往返趟次统计与车辆工作时间的更新
            vehicles_df = vehicles_df[vehicles_df['DRIVE_TIME']<=38]
            vehicles_df = vehicles_df[vehicles_df['WORK_COUNTS']<=9]
                        
            #统计送货顺序
            for route in range(len(route_plan_0)):   #p是线路数
                if len(route_plan_0[route]) > 2:
                    num += 1    
                    for node in route_plan_0[route]:    #index是零售户df索引
                        if node != 0:
                            cust_0['DELIVER_NO'].iloc[node] =  route_plan_0[route].index(node)   #生成配送顺序
                            cust_0['DELIVER_NAME'].iloc[node] = '配送路线{}'.format(num)        #取name中最大的数值
                        
            #遍历完每一条路线后再加入回cust_data里汇总
            cust_data = cust_data.append(cust_0)
        cust_data = cust_data.drop_duplicates(subset = ['CUST_LICENCE_CODE'],keep = 'first')
        
        return cust_data,self.vehicles_df
    
    #多进程
    def multi_process(self):
        pass

    

if __name__ == '__main__':
    time1 = time.time()
    GD = GetData()
    cust_df = GD.Customer()
    vehicles_df = GD.Vehicle()
    
    #dist_mat = Processing().get_matrix(cust_df)
    #cust_route_plan,car_plan = VRP(cust_df,vehicles_df,dist_mat).single_process()
    cust_route_plan,car_plan = VRP(cust_df,vehicles_df).single_process()
    
    # route_plan = OR_Tools(vehicles,vehicles_df,cust_df,distance_matrix,drv_time_mat).main()
    # vehicles_df['ROUTE_PLAN'] = route_plan
    print('\r算法运行总时间：',time.time()-time1)  
    