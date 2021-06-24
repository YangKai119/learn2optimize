from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 用来在图像里面不断添加立方体
def box(ax,x, y, z, dx, dy, dz, color='red'):
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z]*5, **kwargs)#下底
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)#上底
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    return ax
# 显示图形的函数：Items = [[num[0],num[1],num[2],num[3],num[4],num[5],num[6]],]
# Items是N个列表的列表，里面的每个列表数据[放置点O三维坐标，长宽高，颜色]
def show_pic(Items):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.zaxis.set_major_locator(MultipleLocator(50))
    for num in Items:
        box(ax,num[0],num[1],num[2],num[3],num[4],num[5],num[6])
    plt.title('Cube')
    plt.show()
# 根据图显需要的数据，把尺寸数据生成绘图数据的函数
def packaging(O,C,color):
    data = [O[0],O[1],O[2],C[0],C[1],C[2],color]
    return data

