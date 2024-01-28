import csv
import pandas as pd
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


def generate_properties(filename):
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
    # pattern = r'(\d{4}-\d{2})'
    # match = re.search(pattern, filename)
    # date_part = match.group(0)
    location = filename.split('.')[2]
    date_part = filename.split('.')[6]
    print(date_part)

    decimalLatitude = data['decimalLatitude']
    decimalLongitude = data['decimalLongitude']
    # decimal_dict = {'decimalLatitude': decimalLatitude[0], 'decimalLongitude': decimalLongitude[0]}
    elevation = data['elevation']
    # if total_positive != 0 and decimal_dict not in coordinate_list:
    # coordinate_list.append(decimal_dict)
    total_negative = data[data['testResult'] == 'Negative'].shape[0]
    # print("计算总的阳性样本数和阴性样本数")
    # print(f"总的阳性样本数: {total_positive}")
    # print(f"总的阴性样本数: {total_negative}")
    total_rate = total_positive / (total_positive + total_negative)
    # print(f"总的阳性率: {total_rate:.2%}")
    dict_temp = {
        'location': location, 'decimalLatitude': decimalLatitude[0], 'decimalLongitude': decimalLongitude[0],
        'elevation': elevation[0],
        'total_positive': total_positive, 'total_negative': total_negative, 'total_rate': total_rate,
        'date': date_part
    }
    # pathogen_results = data.groupby(['testPathogenName', 'testResult'])['testResult'].count().unstack()
    # print("不同病原体的检测结果正负样本分布")
    # print(pathogen_results)
    # 按病原体计算阳性率
    # pathogen_rate = data.groupby(['testPathogenName'])['testResult'].value_counts(normalize=True)
    # pathogen_rate = pathogen_rate.unstack(level=1)['Positive']
    # # print("按病原体计算阳性率")
    # dict_temp.update(pathogen_rate.to_dict())

    # elif decimal_dict not in coordinate_list:
    #     dict_temp = {
    #         'decimalLatitude': decimalLatitude[0], 'decimalLongitude': decimalLongitude[0], 'elevation': elevation[0],
    #         'total_positive': total_positive, 'total_negative': 0, 'total_rate': 0.0,
    #         'date': date_part
    #     }
    return dict_temp


# # 按testingVialID分组
# grouped = data.groupby('testingVialID')
# # 分析每个样本的阳性率
# for name, group in grouped:
#     positive = group[group['testResult'] == 'Positive'].shape[0]
#     negative = group[group['testResult'] == 'Negative'].shape[0]
#     rate = positive / (positive + negative)
#     print(f"Vial {name}: {rate:.2%} positive")

#
# # 存储到DataFrame中
# vial_rates = pd.DataFrame()
#
# for name, group in grouped:
#     positive = group[group['testResult'] == 'Positive'].shape[0]
#     negative = group[group['testResult'] == 'Negative'].shape[0]
#
#     rate = positive / (positive + negative)
#     vial_rates.loc[name, 'positive'] = positive
#     vial_rates.loc[name, 'negative'] = negative
#     vial_rates.loc[name, 'rate'] = rate
# print("每个样本的阳性率")
# print(vial_rates)


def main():
    folder_to_search = "D:\\edge_download\\NEON_pathogens-mosquito\\NEON_pathogens-mosquito"  # 替换为你要遍历的文件夹路径
    subdirs = list_subdirectories(folder_to_search)
    matching_files = []
    for subdir in subdirs:
        matching = list_files_with_keyword(subdir, "mos_pathogenresults")
        if len(matching) > 0:
            matching_files.append(matching[0])
    # field_names = ['location', 'decimalLatitude', 'decimalLongitude', 'elevation', 'total_positive', 'total_negative',
    #                'total_rate', 'date', 'Dengue virus', 'La Crosse virus', 'Venezuelan equine encephalitis virus',
    #                'Oropouche virus',
    #                'West Nile virus', 'Togaviridae', 'Chikungunya virus', 'Ross River virus',
    #                'Eastern equine encephalitis virus', 'virus', 'Orthobunyavirus sp.',
    #                'Western equine encephalitis virus', 'Japanese encephalitis virus', 'St. Louis encephalitis virus',
    #                'Cowbone Ridge virus', 'California encephalitis virus', 'Flavivirus sp.', 'Alphavirus sp.',
    #                'Flaviviridae', 'Zika virus', 'Mayaro virus', 'Highlands J virus', 'Main Drain virus',
    #                'Yellow fever virus', 'Sindbis virus', 'Una virus', 'Bunyaviridae'
    #                ]
    field_names = ['location', 'decimalLatitude', 'decimalLongitude', 'elevation', 'total_positive', 'total_negative',
                   'total_rate', 'date']
    list_temp = []
    # 1049
    # for file_path in matching_files:
    #     dict_temp, coordinate_list = generate_properties(file_path, coordinate_list)
    #     if dict_temp != None:
    #         list_temp.append(dict_temp)

    for file_path in matching_files:
        dict_temp = generate_properties(file_path)
        if dict_temp != None:
            list_temp.append(dict_temp)
    csv_file = 'rate.csv'
    # 打开CSV文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        # 写入字段名
        writer.writeheader()
        # 写入数据
        for row in list_temp:
            writer.writerow(row)

    #     data = pd.read_csv(file_path)
    #     pathogen_counts = data['testPathogenName'].value_counts()
    #     list_temp.extend(list(pathogen_counts.index))
    #
    # print(set(list_temp))

    #     if pathogen_counts.index.size == 12:
    #         list_temp = list(pathogen_counts.index)
    #         break
    # list_max.extend(list_temp)


if __name__ == "__main__":
    main()
