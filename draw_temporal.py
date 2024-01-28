import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import os
import re


# 遍历文件夹
def list_subdirectories(folder_path):
    subdirectories = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories


def list_files_with_keyword(folder_path, keyword):
    matching_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and keyword in filename:
            matching_files.append(file_path)
    return matching_files


def generate_properties(filename, coordinates):
    latitude = coordinates[0]
    longitude = coordinates[1]
    data = pd.read_csv(filename)
    # # 测试的病原体类型分布情况
    # pathogens = data['testPathogenName'].value_counts()
    # print("病原体类型分布情况")
    # print(pathogens)
    # max_value = pathogens.max()
    # max_indexes = pathogens[pathogens == max_value].index
    # max_values = pathogens[pathogens == max_value]
    # print(f"出现最多的病原体：{list(max_indexes)}")
    # print(f"出现最多的病原体的次数：{list(max_values)}")
    #
    # 不同病原体的检测结果正负样本分布
    # 每个测试vial的测试次数分布
    # test_counts = data.groupby('testingVialID')['testNumber'].max().value_counts()
    # print("每个测试vial的测试次数分布")
    # print(test_counts)
    # 计算总的阳性样本数和阴性样本数
    total_positive = data[data['testResult'] == 'Positive'].shape[0]
    dict_temp = None
    # 使用正则表达式匹配日期模式
    pattern = r'(\d{4}-\d{2})'
    match = re.search(pattern, filename)
    date_part = match.group(0)
    decimalLatitude = data['decimalLatitude']
    decimalLongitude = data['decimalLongitude']
    decimal_temp = np.array([decimalLatitude[0], decimalLongitude[0]])
    elevation = data['elevation']
    if total_positive != 0 and np.array_equiv(coordinates, decimal_temp):
        total_negative = data[data['testResult'] == 'Negative'].shape[0]
        # print("计算总的阳性样本数和阴性样本数")
        # print(f"总的阳性样本数: {total_positive}")
        # print(f"总的阴性样本数: {total_negative}")
        total_rate = total_positive / (total_positive + total_negative)
        # print(f"总的阳性率: {total_rate:.2%}")
        dict_temp = {
            'decimalLatitude': decimalLatitude[0], 'decimalLongitude': decimalLongitude[0], 'elevation': elevation[0],
            'total_positive': total_positive, 'total_negative': total_negative, 'total_rate': total_rate,
            'date': date_part
        }
        # pathogen_results = data.groupby(['testPathogenName', 'testResult'])['testResult'].count().unstack()
        # print("不同病原体的检测结果正负样本分布")
        # print(pathogen_results)
        # 按病原体计算阳性率
        pathogen_rate = data.groupby(['testPathogenName'])['testResult'].value_counts(normalize=True)
        pathogen_rate = pathogen_rate.unstack(level=1)['Positive']
        # print("按病原体计算阳性率")
        dict_temp.update(pathogen_rate.to_dict())
    elif np.array_equiv(coordinates, decimal_temp):
        dict_temp = {
            'decimalLatitude': decimalLatitude[0], 'decimalLongitude': decimalLongitude[0], 'elevation': elevation[0],
            'total_positive': total_positive, 'total_negative': 0, 'total_rate': 0.0,
            'date': date_part
        }
    return dict_temp


def pre_save():
    # 设置Matplotlib的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成你所需的字体
    data = pd.read_csv('data2.csv')
    coordinates = np.empty((3, 2))
    # California
    coordinates[0] = 37.00583, -119.00602
    # Colorado
    coordinates[1] = 40.461894, -103.02929
    # Maryland
    coordinates[2] = 38.890131, -76.560014

    folder_to_search = "D:\\edge_download\\NEON_pathogens-mosquito"  # 替换为你要遍历的文件夹路径
    subdirs = list_subdirectories(folder_to_search)
    matching_files = []
    for subdir in subdirs:
        matching = list_files_with_keyword(subdir, "mos_pathogenresults")
        if len(matching) > 0:
            matching_files.append(matching[0])
    field_names = ['decimalLatitude', 'decimalLongitude', 'elevation', 'total_positive', 'total_negative', 'total_rate',
                   'date',
                   'Dengue virus', 'La Crosse virus', 'Venezuelan equine encephalitis virus', 'Oropouche virus',
                   'West Nile virus', 'Togaviridae', 'Chikungunya virus', 'Ross River virus',
                   'Eastern equine encephalitis virus', 'virus', 'Orthobunyavirus sp.',
                   'Western equine encephalitis virus', 'Japanese encephalitis virus', 'St. Louis encephalitis virus',
                   'Cowbone Ridge virus', 'California encephalitis virus', 'Flavivirus sp.', 'Alphavirus sp.',
                   'Flaviviridae', 'Zika virus', 'Mayaro virus', 'Highlands J virus', 'Main Drain virus',
                   'Yellow fever virus', 'Sindbis virus', 'Una virus', 'Bunyaviridae'
                   ]

    for i in range(3):
        list_temp = []
        # 1049
        for file_path in matching_files:
            dict_temp = generate_properties(file_path, coordinates[i])
            if dict_temp != None:
                list_temp.append(dict_temp)
        csv_file = f'coordinates_data{i}.csv'
        # 打开CSV文件并写入数据
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            # 写入字段名
            writer.writeheader()
            # 写入数据
            for row in list_temp:
                writer.writerow(row)


def draw_line(filename, total_rate, date):
    # 创建折线图
    plt.figure(figsize=(20, 20))
    plt.plot(date, total_rate, marker='o', linestyle='-')
    plt.title("阳性率随时间的变化")
    plt.xlabel("时间（年-月）")
    plt.ylabel("阳性率")
    plt.xticks(rotation=45)
    plt.savefig(f"{filename}.png")
    plt.show()


def draw_bar(filename, total_rate, date):
    # 创建柱状图
    plt.figure(figsize=(20, 20))
    plt.bar(date, total_rate)
    plt.title("不同时间的阳性率比较")
    plt.xlabel("时间（年-月）")
    plt.ylabel("阳性率")
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.savefig(f"{filename}.png")
    plt.show()


def draw():
    # 设置Matplotlib的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成你所需的字体
    for i in range(3):
        filename = f'coordinates_data{i}'
        data = pd.read_csv(f"{filename}.csv")
        total_rate = data['total_rate']
        date = data['date']
        draw_bar(filename, total_rate.values, date.values)


def main():
    # pre_save()
    draw()


if __name__ == "__main__":
    main()
