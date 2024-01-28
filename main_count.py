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


def counts():
    folder_to_search = "D:\\edge_download\\NEON_count-mosquitoes"  # 替换为你要遍历的文件夹路径
    subdirs = list_subdirectories(folder_to_search)
    list_temp = [[] for i in range(len(subdirs))]
    for i in range(len(subdirs)):
        subdir = subdirs[i]
        subdir_split = subdir.split('.')
        location = subdir_split[2]
        date_part = subdir_split[6]
        list_temp[i].append(location)
        list_temp[i].append(date_part)
        matching = list_files_with_keyword(subdir, "mos_expertTaxonomistIDProcessed")
        if len(matching) != 0:
            data = pd.read_csv(matching[0])
            df = data['individualCount']
            individualCounts = np.array(df.to_numpy())
            total_count = np.nansum(individualCounts)
            list_temp[i].append(total_count)
        else:
            list_temp[i].append(0.0)
    field_names = ['location', 'date', 'total_count']
    list_temp.insert(0, field_names)
    csv_file = 'count.csv'
    # 打开CSV文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list_temp)


def main():
    counts()


if __name__ == "__main__":
    main()
