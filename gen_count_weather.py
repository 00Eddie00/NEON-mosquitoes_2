import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager


def gen_relation_count():
    counts = pd.read_csv('count.csv')
    weather = pd.read_csv('weather.csv')

    counts_location = np.array(counts['location'].to_numpy())
    weather_location = np.array(weather['location'].to_numpy())

    counts_date = np.array(counts['date'].to_numpy())
    weather_date = np.array(weather['date'].to_numpy())

    # 使用 numpy.core.defchararray.add 进行拼接
    counts_result = np.char.add(counts_location.astype('U'), counts_date.astype('U'))
    weather_result = np.char.add(weather_location.astype('U'), weather_date.astype('U'))

    # 找到两个数组中相同元素的下标
    common_indices_array1 = np.where(np.isin(counts_result, weather_result))[0]  # count
    common_indices_array2 = np.where(np.isin(weather_result, counts_result))[0]  # weather

    field_names = ['location', 'date', 'decimalLatitude', 'decimalLongitude', "mean_wind_speed_1",
                   "mean_wind_speed_5",
                   "mean_wind_speed_16", "mean_wind_speed_22", "mean_wind_speed_30", "mean_pres", "mean_humid",
                   "mean_shortRad", "mean_temp", "mean_precip", 'total_count']
    list_temp = [[] for i in range(len(common_indices_array2) + 1)]
    list_temp[0] = field_names
    for i in range(len(common_indices_array2)):
        counts_i = common_indices_array1[i]
        counts_selected_data = counts.iloc[counts_i]
        counts_selected_array = np.array(counts_selected_data)

        weather_i = common_indices_array2[i]
        weather_selected_data = weather.iloc[weather_i]
        weather_selected_array = np.array(weather_selected_data)

        temp = np.concatenate((weather_selected_array, [counts_selected_array[2]]))

        list_temp[i + 1] = temp.tolist()
    csv_file = 'count_weather.csv'
    # 打开CSV文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list_temp)


def gen_relation_rate():
    rate = pd.read_csv('rate.csv')
    weather = pd.read_csv('weather.csv')

    counts_location = np.array(rate['location'].to_numpy())
    weather_location = np.array(weather['location'].to_numpy())

    counts_date = np.array(rate['date'].to_numpy())
    weather_date = np.array(weather['date'].to_numpy())

    # 使用 numpy.core.defchararray.add 进行拼接
    counts_result = np.char.add(counts_location.astype('U'), counts_date.astype('U'))
    weather_result = np.char.add(weather_location.astype('U'), weather_date.astype('U'))

    # 找到两个数组中相同元素的下标
    common_indices_array1 = np.where(np.isin(counts_result, weather_result))[0]  # count
    common_indices_array2 = np.where(np.isin(weather_result, counts_result))[0]  # weather
    field_names = ['location', 'date', 'decimalLatitude', 'decimalLongitude', "mean_wind_speed_1",
                   "mean_wind_speed_5",
                   "mean_wind_speed_16", "mean_wind_speed_22", "mean_wind_speed_30", "mean_pres", "mean_humid",
                   "mean_shortRad", "mean_temp", "mean_precip", 'rate']
    list_temp = [[] for i in range(len(common_indices_array2) + 1)]
    list_temp[0] = field_names
    for i in range(len(common_indices_array2)):
        rates_i = common_indices_array1[i]
        rates_selected_data = rate.iloc[rates_i]
        rates_selected_array = np.array(rates_selected_data)

        weather_i = common_indices_array2[i]
        weather_selected_data = weather.iloc[weather_i]
        weather_selected_array = np.array(weather_selected_data)

        temp = np.concatenate((weather_selected_array, [rates_selected_array[6]]))

        list_temp[i + 1] = temp.tolist()
    csv_file = 'rate_weather.csv'
    # 打开CSV文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list_temp)



def main():
    point = 'KONZ'
    count_weather = "count_weather.csv"  # 替换为实际的CSV文件路径
    gen_relation_rate()


if __name__ == "__main__":
    main()
