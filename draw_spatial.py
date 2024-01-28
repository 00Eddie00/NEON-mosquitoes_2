import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


def draw_coordinate(decimalLatitude_list, decimalLongitude_list, data_list, fig_name):
    # 创建一个地图
    plt.figure(figsize=(10, 10))
    m = Basemap(
        projection='merc',  # 投影方式，可以根据需要选择不同的投影方式
        llcrnrlat=decimalLatitude_list.min() - 1, urcrnrlat=decimalLatitude_list.max() + 1,
        llcrnrlon=decimalLongitude_list.min() - 1, urcrnrlon=decimalLongitude_list.max() + 1,
        resolution='i'  # 分辨率，'i' 表示中等分辨率
    )
    # 绘制地图边界、海岸线和国家边界
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='white')
    # 绘制州的边界线
    m.drawstates()

    # 定义颜色映射范围，使用高对比度的颜色映射（可以根据需要选择其他颜色映射）
    color_map = plt.cm.get_cmap('rainbow')

    # 绘制散点图，根据经度和纬度来定位点
    x, y = m(decimalLongitude_list.values, decimalLatitude_list.values)
    scatter = plt.scatter(x, y, c=data_list, cmap=color_map, s=20, edgecolor='k', alpha=0.7)
    name_list = fig_name.split('_')
    p_name = ''.join(name_list[2:])

    # 添加颜色条
    plt.colorbar(scatter, label=p_name)

    # 添加标题和标签
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.savefig(f"{fig_name}.png")

    # 显示图形
    plt.show()


def extract_nz():
    count_rate = pd.read_csv('count_rate.csv')
    total_rate = np.array(count_rate['total_rate'].to_numpy())
    indices = np.nonzero(total_rate)
    temp = np.array(count_rate.to_numpy())[indices]
    return temp


def extract_ez():
    count_rate = pd.read_csv('count_rate.csv')
    total_rate = np.array(count_rate['total_rate'].to_numpy())
    indices = np.where(total_rate == 0 )
    temp = np.array(count_rate.to_numpy())[indices]
    return temp


def main():
    temp_nz = extract_nz()
    location_nz = temp_nz[:, 0]
    decimalLatitude_list_nz = temp_nz[:, 2]
    decimalLongitude_list_nz = temp_nz[:, 3]

    temp_ez = extract_ez()
    location_ez = temp_ez[:, 0]
    decimalLatitude_list_ez = temp_ez[:, 2]
    decimalLongitude_list_ez = temp_ez[:, 3]

    # filed_names = ['WOOD', 'UNDE', 'KONZ', 'CPER', 'BONA', 'TALL', 'OSBS']
    # 设置Matplotlib的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成所需的字体
    plt.figure(figsize=(10, 10))
    m = Basemap(
        projection='merc',  # 投影方式，可以根据需要选择不同的投影方式
        llcrnrlat=decimalLatitude_list_ez.min() - 1, urcrnrlat=decimalLatitude_list_ez.max() + 1,
        llcrnrlon=decimalLongitude_list_ez.min() - 1, urcrnrlon=decimalLongitude_list_ez.max() + 1,
        resolution='i'  # 分辨率，'i' 表示中等分辨率
    )

    # 在地图上绘制经纬度点
    x_ez, y_ez = m(decimalLongitude_list_ez, decimalLatitude_list_ez)
    x_nz, y_nz = m(decimalLongitude_list_nz, decimalLatitude_list_nz)
    m.scatter(x_ez, y_ez, marker='o', color='g', zorder=5)
    m.scatter(x_nz, y_nz, marker='o', color='r', zorder=5)


    # 标注地址名称
    # for i, label in enumerate(location_nz):
    #     plt.text(x_nz[i], y_nz[i], label, color='b', fontsize=8, ha='left', va='bottom', zorder=6)

        # 标注地址名称
    for i, label in enumerate(location_ez):
        plt.text(x_ez[i], y_ez[i], label, color='b', fontsize=8, ha='left', va='bottom', zorder=6)

    # 绘制地图边界、海岸线和国家边界
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='white')
    # 绘制州的边界线
    m.drawstates()
    plt.savefig("map")


if __name__ == "__main__":
    main()
