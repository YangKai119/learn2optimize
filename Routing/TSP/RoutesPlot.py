import matplotlib.pyplot as plt
import numpy as np


def DrawPointMap(sol, name):
    plt.figure()
    depot = sol.nodes[0]
    x0 = depot.x
    y0 = depot.y
    x = [sol.nodes[i].x for i in range(len(sol.nodes))]  # 所有节点的x集合
    y = [sol.nodes[i].y for i in range(len(sol.nodes))]  # 所有节点的y集合
    routes = []
    rou = np.array(sol.route)
    zero_idx = np.where(rou == 0)[0]
    for i in range(len(zero_idx) - 1):
        routes.append(list(rou[zero_idx[i]:zero_idx[i + 1] + 1]))   # 掐头去尾
    vehicles = len(routes)
    plt.scatter(x, y, color = 'k', marker = 'o'      # 要标记的点的坐标、大小及颜色
                ,label = 'Customer')
    # annotate an important value
    plt.scatter(x0, y0, s = 200, color = 'r',
                marker='*',label = 'Depot')          # 要标记的点的坐标、大小及颜色
    # 色卡，后期可以存入sql数据库
    colors = ['darkkhaki','darkmagenta','darkolivegreen','hotpink','indianred','indigo','orange','khaki','lavender','lavenderblush','lawngreen','lemonchiffon']
    for idx in range(vehicles):
        route(x0, y0, x, y, routes[idx], colors[idx])

    plt.title("{0} obj val: {1}".format(name, sol.tot_dist))
    plt.legend()             # 图例
    plt.show()

# 配送路线
def route(x0, y0, x, y, rou ,color):
    print(rou)
    for i in range(len(rou)-1):
        begin_idx = rou[i]
        end_idx = rou[i+1]
        # if i == 0:               # 不能使用if elif的结构，因为不是分支的关系
        x_begin,y_begin = x[begin_idx], y[begin_idx]
        x_end, y_end = x[end_idx], y[end_idx]
        plt_arrow(x_begin, y_begin, x_end, y_end, color)
        # if i == len(rou) - 1:
        #     x_begin, y_begin = x[idx], y[idx]
        #     x_end, y_end = x0, y0
        # if i < len(rou) - 1:
        #     print(i)
        #     print(rou[i+1])
        #     x_begin, y_begin = x[idx], y[idx]
        #     x_end, y_end = x[rou[i+1]], y[rou[i+1]]
        #
        # plt_arrow(x_begin, y_begin, x_end, y_end, color)

def plt_arrow(x_begin, y_begin, x_end, y_end, color):

    plt.arrow(x_begin,
              y_begin,
              x_end - x_begin,
              y_end - y_begin,
              length_includes_head=True,     # 增加的长度包含箭头部分
              head_width = 0.02,
              head_length =0.02,
              fc=color,
              ec=color)

if __name__=='__main__':

    DrawPointMap()
