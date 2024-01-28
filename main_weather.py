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


def weather():
    folder_to_search = "D:\\edge_download\\NEON_weather-stats"  # 替换为你要遍历的文件夹路径
    subdirs = list_subdirectories(folder_to_search)
    # 不同高度的温度（1 5 16 22 30）、气压、湿度、辐射、温度、降水量
    key_words = ["010.01D.wss_daily_wind", "020.01D.wss_daily_wind", "030.01D.wss_daily_wind", "040.01D.wss_daily_wind",
                 "050.01D.wss_daily_wind", "wss_daily_pres", "wss_daily_humid", "wss_daily_shortRad", "wss_daily_temp",
                 "wss_daily_precip"]
    matching_files = [[] for i in range(len(subdirs))]

    for i in range(len(subdirs)):
        subdir = subdirs[i]
        for j in range(len(key_words)):
            kw = key_words[j]
            matching = list_files_with_keyword(subdir, kw)

            if len(matching) > 0:
                matching_files[i].append(matching[0])
            else:
                matching_files[i].append(None)

    list_temp = [[] for i in range(len(subdirs))]
    for i in range(len(list_temp)):
        for j in range(len(key_words)):
            file_name = matching_files[i][j]
            if file_name is not None:
                # 读取 CSV 文件
                df = pd.read_csv(file_name)
                # 获取第二列数据，不包括第一行
                second_column_data = df.iloc[1:, 1].tolist()
                average_value_without_nan = np.nanmean(second_column_data)
                list_temp[i].append(average_value_without_nan)
            else:
                list_temp[i].append('')

    for i in range(len(subdirs)):
        subdir = subdirs[i]
        location = subdir.split('.')[2]

        matching = list_files_with_keyword(subdir, "sensor_positions")
        # 读取 CSV 文件
        df = pd.read_csv(matching[0], header=None)  # 没有列标签

        # 指定要选择的行和列的标签
        row_index = 1  # 选择第3行（索引从0开始）
        column_index_to_select = [15, 16]  # 替换为你要选择的列的索引
        # 选择指定行和列的数据
        selected_data = df.iloc[row_index, column_index_to_select]
        selected_array = np.array(selected_data)
        referenceLatitude = selected_array[0]
        referenceLongitude = selected_array[1]
        # 使用正则表达式匹配日期模式
        pattern = r'(\d{4}-\d{2})'
        match = re.search(pattern, matching[0])
        date_part = match.group(0)
        # print(f"referenceLatitude:{referenceLatitude},referenceLongitude:{referenceLongitude},date_part:{date_part}")
        list_temp[i].insert(0, referenceLongitude)
        list_temp[i].insert(0, referenceLatitude)
        list_temp[i].insert(0, date_part)
        list_temp[i].insert(0, location)
    field_names = ['location', 'date', 'decimalLatitude', 'decimalLongitude', "mean_wind_speed_1", "mean_wind_speed_5",
                   "mean_wind_speed_16", "mean_wind_speed_22", "mean_wind_speed_30", "mean_pres", "mean_humid",
                   "mean_shortRad", "mean_temp", "mean_precip"]
    list_temp.insert(0, field_names)
    csv_file = 'weather.csv'
    # 打开CSV文件并写入数据
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(list_temp)


def main():
    weather()


if __name__ == "__main__":
    main()
