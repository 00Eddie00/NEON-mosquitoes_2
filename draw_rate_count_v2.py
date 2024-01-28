import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager
from function_rate_count import gen_linear_regression_count_rate


def gen_relation_count_rate():
    counts = pd.read_csv('count_weather.csv')
    rate = pd.read_csv('rate_weather.csv')

    counts_location = np.array(counts['location'].to_numpy())
    rate_location = np.array(rate['location'].to_numpy())

    counts_date = np.array(counts['date'].to_numpy())
    rate_date = np.array(rate['date'].to_numpy())

    # 使用 numpy.core.defchararray.add 进行拼接
    counts_result = np.char.add(counts_location.astype('U'), counts_date.astype('U'))
    rate_result = np.char.add(rate_location.astype('U'), rate_date.astype('U'))

    # 找到两个数组中相同元素的下标
    common_indices_array1 = np.where(np.isin(counts_result, rate_result))[0]  # count
    common_indices_array2 = np.where(np.isin(rate_result, counts_result))[0]  # rate

    field_names = ['location', 'date', 'decimalLatitude', 'decimalLongitude', "mean_wind_speed_1",
                   "mean_wind_speed_5",
                   "mean_wind_speed_16", "mean_wind_speed_22", "mean_wind_speed_30", "mean_pres", "mean_humid",
                   "mean_shortRad", "mean_temp", "mean_precip", 'total_count', 'total_rate']
    list_temp = [[] for i in range(len(common_indices_array2) + 1)]
    list_temp[0] = field_names
    for i in range(len(common_indices_array2)):
        counts_i = common_indices_array1[i]
        counts_selected_data = counts.iloc[counts_i]
        counts_selected_array = np.array(counts_selected_data)

        rate_i = common_indices_array2[i]
        rate_selected_data = rate.iloc[rate_i]
        rate_selected_array = np.array(rate_selected_data)

        temp = np.concatenate((counts_selected_array, [rate_selected_array[-1]]))

        list_temp[i + 1] = temp.tolist()
    csv_file = 'count_rate.csv'
    # 打开CSV文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list_temp)


def find_p():
    count_rate = pd.read_csv('count_rate.csv')
    location = np.array(count_rate['location'].to_numpy())
    total_rate = np.array(count_rate['total_rate'].to_numpy())
    indices = np.nonzero(total_rate)
    print(location[indices])


def draw_count(point, csv_file_path):
    rate_coefficients, rate_intercept, count_coefficients, count_intercept = gen_linear_regression_count_rate()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成你所需的字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    df = pd.read_csv(csv_file_path)

    selected_data = df[df['location'] == point]

    # 提取列数据
    time = selected_data['date']
    wind_speed = selected_data['mean_wind_speed_5']
    humidity = selected_data['mean_humid']
    pressure = selected_data['mean_pres']
    precipitation = selected_data['mean_precip']
    temperature = selected_data['mean_temp']
    mosquito_count = selected_data['total_count']

    # 设置画布大小
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制左侧纵坐标轴（蚊子数量）
    ax1.plot(time, mosquito_count, 'b-', label='Mosquito Count', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_xticklabels(time, rotation=50, fontsize=8)

    ax1.set_ylabel('Mosquito Count', color='b')
    ax1.tick_params('y', colors='b')

    # 创建右侧纵坐标轴
    # ax2 = ax1.twinx()
    #
    # # 绘制右侧纵坐标轴（温度）
    # ax2.plot(time, temperature.fillna(0.0), 'r-', label='Temperature')
    # ax2.set_ylabel('Temperature (°C)', color='r')
    # ax2.tick_params('y', colors='r', labelsize=7)

    # 创建第二个右侧纵坐标轴
    ax3 = ax1.twinx()

    # 位置调整，使得第二个右侧纵坐标轴共享横坐标轴
    # ax3.spines['right'].set_position(('outward', 35))

    # 绘制右侧纵坐标轴（湿度）
    ax3.plot(time, humidity.fillna(0.0), 'g-', label='Humidity')
    ax3.set_ylabel('Humidity (%)', color='g')
    ax3.tick_params('y', colors='g', labelsize=7)

    # 创建第三个右侧纵坐标轴
    ax4 = ax1.twinx()

    # 位置调整，使得第三个右侧纵坐标轴共享横坐标轴
    ax4.spines['right'].set_position(('outward', 35))

    # 绘制右侧纵坐标轴（气压）
    ax4.plot(time, pressure.fillna(0.0), 'm-', label='Pressure')
    ax4.set_ylabel('Pressure (kPa)', color='m')
    ax4.tick_params('y', colors='m', labelsize=7)

    # 创建第四个右侧纵坐标轴
    # ax5 = ax1.twinx()
    #
    # # 位置调整，使得第四个右侧纵坐标轴共享横坐标轴
    # ax5.spines['right'].set_position(('outward', 120))
    #
    # # 绘制右侧纵坐标轴（降水）
    # ax5.plot(time, precipitation.fillna(0.0), 'c-', label='Precipitation')
    # ax5.set_ylabel('Precipitation (mm)', color='c')
    # ax5.tick_params('y', colors='c', labelsize=7)

    # 创建第五个右侧纵坐标轴
    ax6 = ax1.twinx()

    # 位置调整，使得第五个右侧纵坐标轴共享横坐标轴
    ax6.spines['right'].set_position(('outward', 75))

    # 绘制右侧纵坐标轴（风速）
    ax6.plot(time, wind_speed.fillna(0.0), 'y-', label='wind_speed')
    ax6.set_ylabel('wind_speed (m/s)', color='y')
    ax6.tick_params('y', colors='y', labelsize=7)

    # 显示图例
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    equation_text = f'y={count_coefficients[0]}*x_temp+({count_coefficients[1]})*x_humid+({count_coefficients[2]})*x_pres+({count_coefficients[3]})*x_precip+({count_coefficients[4]})*x_ws+({count_intercept})'  # 你可以修改为你需要显示的方程
    ax1.text(0.5, -0.14, equation_text, transform=ax1.transAxes, ha='center', fontsize=8)

    plt.savefig(f"count/{point}.png")

    # 显示图形
    # plt.show()


def draw_rate(point, csv_file_path):
    rate_coefficients, rate_intercept, count_coefficients, count_intercept = gen_linear_regression_count_rate()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换成你所需的字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    df = pd.read_csv(csv_file_path)

    # 假设你要选择湿度大于80的数据
    selected_data = df[df['location'] == point]

    # 提取列数据
    time = selected_data['date']
    wind_speed = selected_data['mean_wind_speed_5']
    humidity = selected_data['mean_humid']
    pressure = selected_data['mean_pres']
    precipitation = selected_data['mean_precip']
    temperature = selected_data['mean_temp']
    rate = selected_data['total_rate']

    # 设置画布大小
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制左侧纵坐标轴（蚊子数量）
    ax1.plot(time, rate * 100, 'b-', label='Positive Rate', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_xticklabels(time, rotation=50, fontsize=8)

    ax1.set_ylabel('Positive Rate (%)', color='b')
    ax1.tick_params('y', colors='b')

    # 创建右侧纵坐标轴
    ax2 = ax1.twinx()

    # 绘制右侧纵坐标轴（温度）
    ax2.plot(time, temperature.fillna(0.0), 'r-', label='Temperature')
    ax2.set_ylabel('Temperature (°C)', color='r')
    ax2.tick_params('y', colors='r', labelsize=7)

    # 创建第二个右侧纵坐标轴
    ax3 = ax1.twinx()

    # 位置调整，使得第二个右侧纵坐标轴共享横坐标轴
    ax3.spines['right'].set_position(('outward', 35))

    # 绘制右侧纵坐标轴（湿度）
    ax3.plot(time, humidity.fillna(0.0), 'g-', label='Humidity')
    ax3.set_ylabel('Humidity (%)', color='g')
    ax3.tick_params('y', colors='g', labelsize=7)

    # # 创建第三个右侧纵坐标轴
    # ax4 = ax1.twinx()
    #
    # # 位置调整，使得第三个右侧纵坐标轴共享横坐标轴
    # ax4.spines['right'].set_position(('outward', 75))
    #
    # # 绘制右侧纵坐标轴（气压）
    # ax4.plot(time, pressure.fillna(0.0), 'm-', label='Pressure')
    # ax4.set_ylabel('Pressure (kPa)', color='m')
    # ax4.tick_params('y', colors='m', labelsize=7)
    #
    # # 创建第四个右侧纵坐标轴
    # ax5 = ax1.twinx()
    #
    # # 位置调整，使得第四个右侧纵坐标轴共享横坐标轴
    # ax5.spines['right'].set_position(('outward', 120))
    #
    # # 绘制右侧纵坐标轴（降水）
    # ax5.plot(time, precipitation.fillna(0.0), 'c-', label='Precipitation')
    # ax5.set_ylabel('Precipitation (mm)', color='c')
    # ax5.tick_params('y', colors='c', labelsize=7)

    # 创建第五个右侧纵坐标轴
    ax6 = ax1.twinx()

    # 位置调整，使得第五个右侧纵坐标轴共享横坐标轴
    ax6.spines['right'].set_position(('outward', 85))

    # 绘制右侧纵坐标轴（辐射）
    ax6.plot(time, wind_speed.fillna(0.0), 'y-', label='wind_speed')
    ax6.set_ylabel('wind_speed (m/s)', color='y')
    ax6.tick_params('y', colors='y', labelsize=7)

    # 显示图例
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    equation_text = f'y={rate_coefficients[0]}*x_temp+({rate_coefficients[1]})*x_humid+({rate_coefficients[2]})*x_pres+({rate_coefficients[3]})*x_precip+({rate_coefficients[4]})*x_ws+({rate_intercept})'  # 你可以修改为你需要显示的方程
    ax1.text(0.5, -0.14, equation_text, transform=ax1.transAxes, ha='center', fontsize=8)

    plt.savefig(f"rate/{point}.png")

    # 显示图形
    # plt.show()


def main():
    points = ['WOOD', 'UNDE', 'KONZ', 'CPER', 'BONA', 'TALL', 'OSBS']
    file_name = "count_rate.csv"
    # gen_relation_count_rate()
    for point in points:
        draw_count(point, file_name)
        draw_rate(point, file_name)

    # WOOD UNDE KONZ CPER中
    # BONA西
    # TALL OSBS东


if __name__ == "__main__":
    main()
