import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def linear_regression_summary(temperature, humidity, pressure, precipitation, wind_speed, result):
    # Define Feature Names: A list named 'feature_names' is defined, containing the names of features used for the regression analysis.
    feature_names = ["Temperature", "Humidity", "Pressure", "Precipitation", "Wind Speed"]
    # Construct Feature Matrix: The temperature, humidity, pressure, precipitation, and wind speed are merged into a feature matrix 'X' using NumPy's 'column_stack' function.
    X = np.column_stack((temperature, humidity, pressure, precipitation, wind_speed))
    # Standardize Features: The features in matrix 'X' are standardized using 'StandardScaler', resulting in 'X_standardized'.
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    # Create Linear Regression Model: An instance of the 'LinearRegression' model is initialized.
    model = LinearRegression()
    # Fit the Model: The linear regression model is fitted using the standardized feature matrix 'X_standardized' and the target variable 'result'.
    model.fit(X_standardized, result)
    # Add Intercept Column: The 'add_constant' function from 'statsmodels' is used to add an intercept column to 'X_standardized', resulting in 'X_with_intercept'.
    X_with_intercept = sm.add_constant(X_standardized)
    df = pd.DataFrame(X_with_intercept, columns=["const"] + feature_names)
    # Create OLS Model: A linear regression model is created using the 'OLS' method from 'statsmodels'.
    model_with_intercept = sm.OLS(result, df)
    # Fit the Model: The 'OLS' model is fitted using the data with the intercept term added.
    results = model_with_intercept.fit()
    # Output Model Summary: The summary statistics of the model are printed.
    print(results.summary())


def rate_func_no_sdr(windspeed, disease_positive_rate):
    # Prepare Data: The function takes two input arrays, 'windspeed' and 'disease_positive_rate', and combines them into a feature matrix 'X' and a target array 'y'.
    X = np.column_stack((windspeed,))
    y = disease_positive_rate
    # Create and Fit Model: A linear regression model is created using 'LinearRegression' from scikit-learn and then fitted with the feature matrix X and target array y.
    model_standardized = LinearRegression()
    model_standardized.fit(X, y)
    # Retrieve Coefficients and Intercept: After fitting the model, the regression coefficients ('coefficients_standardized') and intercept ('intercept_standardized') are obtained.
    coefficients_standardized = model_standardized.coef_
    intercept_standardized = model_standardized.intercept_
    # Make Predictions: The model is used to make predictions ('y_pred') based on the input 'windspeed'.
    y_pred = model_standardized.predict(X)
    # Visualize Data and Model: The function plots a scatter plot of the actual data points ('X' vs 'y') and overlays the fitted regression line ('X' vs 'y_pred'). Additionally, the function annotates the plot with the regression equation, which includes the intercept and coefficient values.
    plt.scatter(X, y, label='actual value', alpha=0.5)
    plt.plot(X, y_pred, label='fitted line', color='red')
    # Add regression equation information
    equation_text = f'rate = {intercept_standardized:.4f} + {coefficients_standardized[0]:.4f} * wind_speed'
    plt.text(0.55, 0.9, equation_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
             color='blue')
    # Set the graphic title and label
    plt.title('Relationship between wind speed and positivity rate')
    plt.xlabel('wind speed')
    plt.ylabel('positivity rate')
    # Add legend
    plt.legend()
    plt.savefig("rate/wind_speed_no_sdr")


def count_func_humidity_no_sdr(humidity, mosquito_count):
    indexs = np.where(humidity == 0.0)[0]
    humidity = np.delete(humidity, indexs)
    mosquito_count = np.delete(mosquito_count, indexs)

    # 假设 X 是包含所有自变量的数组
    X = np.column_stack((humidity,))
    y = mosquito_count

    # 创建并拟合模型
    model_standardized = LinearRegression()
    model_standardized.fit(X, y)

    # 输出标准化后的回归系数
    coefficients_standardized = model_standardized.coef_
    intercept_standardized = model_standardized.intercept_

    # 预测值
    y_pred = model_standardized.predict(X)

    # 绘制散点图和拟合直线
    plt.scatter(X, y, label='actual value', alpha=0.5)
    plt.plot(X, y_pred, label='fitted line', color='red')

    # 添加回归方程信息
    equation_text = f'm_p = {intercept_standardized:.4f} +({coefficients_standardized[0]:.4f} * humidity)'
    plt.text(0.4, 0.9, equation_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
             color='blue')

    # 设置图形标题和标签
    plt.title('Relationship between humidity and mosquito populations')
    plt.xlabel('humidity')
    plt.ylabel('mosquito populations')

    # 添加图例
    plt.legend()
    plt.savefig("count/humidity_no_sdr")

    # 显示图形
    plt.show()


def count_func_pressure_no_sdr(pressure, mosquito_count):
    indexs = np.where(pressure < 85.0)[0]
    pressure = np.delete(pressure, indexs)
    mosquito_count = np.delete(mosquito_count, indexs)

    # 假设 X 是包含所有自变量的数组
    X = np.column_stack((pressure,))
    y = mosquito_count

    # 创建并拟合模型
    model_standardized = LinearRegression()
    model_standardized.fit(X, y)

    # 输出标准化后的回归系数
    coefficients_standardized = model_standardized.coef_
    intercept_standardized = model_standardized.intercept_

    # 预测值
    y_pred = model_standardized.predict(X)

    # 绘制散点图和拟合直线
    plt.scatter(X, y, label='actual value', alpha=0.5)
    plt.plot(X, y_pred, label='fitted line', color='red')

    # 添加回归方程信息
    equation_text = f'm_p = {intercept_standardized:.4f} +({coefficients_standardized[0]:.4f} * pressure)'
    plt.text(0.4, 0.9, equation_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
             color='blue')

    # 设置图形标题和标签
    plt.title('Relationship between pressure and mosquito populations')
    plt.xlabel('pressure')
    plt.ylabel('mosquito populations')

    # 添加图例
    plt.legend()
    plt.savefig("count/pressure_no_sdr")

    # 显示图形
    plt.show()


def count_func_windspeed_no_sdr(wind_speed, mosquito_count):
    # 假设 X 是包含所有自变量的数组
    X = np.column_stack((wind_speed,))
    y = mosquito_count

    # 创建并拟合模型
    model_standardized = LinearRegression()
    model_standardized.fit(X, y)

    # 输出标准化后的回归系数
    coefficients_standardized = model_standardized.coef_
    intercept_standardized = model_standardized.intercept_

    # 预测值
    y_pred = model_standardized.predict(X)

    # 绘制散点图和拟合直线
    plt.scatter(X, y, label='actual value', alpha=0.5)
    plt.plot(X, y_pred, label='fitted line', color='red')

    # 添加回归方程信息
    equation_text = f'm_p = {intercept_standardized:.4f} +({coefficients_standardized[0]:.4f} * wind_speed)'
    plt.text(0.4, 0.9, equation_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
             color='blue')

    # 设置图形标题和标签
    plt.title('Relationship between wind speed and mosquito populations')
    plt.xlabel('wind speed')
    plt.ylabel('mosquito populations')

    # 添加图例
    plt.legend()
    plt.savefig("count/wind_speed_no_sdr")

    # 显示图形
    plt.show()


# Extracts specific columns' data into NumPy arrays.
# Any NaN values in these arrays are replaced with zeros using ‘np.nan_to_num’.
def np_list(selected_data):
    wind_speed = np.array(selected_data['mean_wind_speed_5'].to_numpy())
    humidity = np.array(selected_data['mean_humid'].to_numpy())
    pressure = np.array(selected_data['mean_pres'].to_numpy())
    precipitation = np.array(selected_data['mean_precip'].to_numpy())
    temperature = np.array(selected_data['mean_temp'].to_numpy())
    rate = np.array(selected_data['total_rate'].to_numpy())
    count = np.array(selected_data['total_count'].to_numpy())

    wind_speed = np.nan_to_num(wind_speed, nan=0)
    humidity = np.nan_to_num(humidity, nan=0)
    pressure = np.nan_to_num(pressure, nan=0)
    precipitation = np.nan_to_num(precipitation, nan=0)
    temperature = np.nan_to_num(temperature, nan=0)
    rate = np.nan_to_num(rate, nan=0)
    count = np.nan_to_num(count, nan=0)
    return wind_speed, humidity, pressure, precipitation, temperature, rate, count


def season_count(df):
    selected_data = df[df['total_count'] != 0]
    date = np.array(selected_data['date'].to_numpy())
    count = np.array(selected_data['total_count'].to_numpy())

    sum_monthly = np.zeros((12))

    for i in range(len(date)):
        mm = int(date[i].split('/')[1])
        sum_monthly[mm - 1] = sum_monthly[mm - 1] + count[i]

    # 使用matplotlib的bar函数绘制柱状图
    bars = plt.bar(range(1, len(sum_monthly) + 1), sum_monthly)
    # 在每个柱子的上方显示值
    # for i, bar in enumerate(bars):
    #     yval = bar.get_height()
    # plt.text(bar.get_x() + bar.get_width() / 2, yval, str(int(yval)),
    #          ha='center', va='bottom', fontsize=8, color='black')
    # 设置横坐标标签位置
    plt.xticks(np.arange(1, 13), np.arange(1, 13))
    # 添加标签和标题
    plt.xlabel('month')
    plt.ylabel('population')

    plt.savefig("count/season_count")


def season_rate(df):
    selected_data = df[df['total_rate'] != 0]
    date = np.array(selected_data['date'].to_numpy())
    rate = np.array(selected_data['total_rate'].to_numpy())

    sum_monthly = np.zeros((12, 17))
    index = np.zeros((12))
    new_sum = np.zeros((12))
    for i in range(len(date)):
        mm = int(date[i].split('/')[1])
        j = int(index[mm - 1])
        sum_monthly[mm - 1, j] = rate[i]
        index[mm - 1] = j + 1
    for s in range(len(sum_monthly)):
        new_sum[s] = np.max(sum_monthly[s])

    # 使用matplotlib的bar函数绘制柱状图
    bars = plt.bar(range(1, len(new_sum) + 1), new_sum)

    # 在每个柱子的上方显示值
    # for i, bar in enumerate(bars):
    #     yval = bar.get_height()
    # plt.text(bar.get_x() + bar.get_width() / 2, yval, str(int(yval)),
    #          ha='center', va='bottom', fontsize=8, color='black')
    def percentage_formatter(x, pos):
        return f'{x:.2%}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # 设置横坐标标签位置
    plt.xticks(np.arange(1, 13), np.arange(1, 13))
    # # 添加标签和标题
    plt.xlabel('month')
    plt.ylabel('positive rate')
    plt.legend()

    # 移除右上角的图例
    plt.legend().set_visible(False)
    plt.savefig("rate/season_rate")


def latitude_count(df):
    selected_data = df[df['total_count'] != 0]
    latitude = np.array(selected_data['decimalLongitude'].to_numpy())
    count = np.array(selected_data['total_count'].to_numpy())
    unique_latitude = np.unique(latitude)
    length = len(unique_latitude)
    sum_latitude = np.zeros((length))
    for i in range(length):
        l = unique_latitude[i]
        index = np.where(latitude == l)
        temp_count = count[index]
        sum_latitude[i] = np.sum(temp_count)
    bars = plt.bar(unique_latitude, sum_latitude)
    # plt.xticks(np.arange(1, 13), np.arange(1, 13))

    # 添加标签和标题
    plt.xlabel('longitude')
    plt.ylabel('population')

    plt.savefig("count/longitude_count")


def latitude_rate(df):
    selected_data = df[df['total_rate'] != 0]
    latitude = np.array(selected_data['decimalLongitude'].to_numpy())
    rate = np.array(selected_data['total_rate'].to_numpy())
    unique_latitude = np.unique(latitude)
    length = len(unique_latitude)  # 7
    sum_latitude = np.zeros((length, 7))
    new_sum = np.zeros((length))
    for i in range(length):
        l = unique_latitude[i]
        index = np.where(latitude == l)
        temp_count = rate[index]
        sum_latitude[i, :len(temp_count)] = temp_count

    for s in range(len(sum_latitude)):
        new_sum[s] = np.max(sum_latitude[s])
        # 使用matplotlib的bar函数绘制柱状图
    bars = plt.bar(range(1, len(new_sum) + 1), new_sum)

    # 添加标签和标题
    plt.xlabel('longitude')
    plt.ylabel('positive rate')
    plt.legend()

    def percentage_formatter(x, pos):
        return f'{x:.2%}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # 设置横坐标标签位置及格式
    plt.xticks(np.arange(1, new_sum.shape[0] + 1),
               [f'{val:.3f}' for val in unique_latitude])

    # 移除右上角的图例
    plt.legend().set_visible(False)
    plt.savefig("rate/longitude_rate")


def f_a_count(df):
    TOOL = df[df['location'] == 'TOOL']  # 北
    tool_date = np.array(TOOL['date'].to_numpy())
    tool_count = np.array(TOOL['total_count'].to_numpy())
    OSBS = df[df['location'] == 'OSBS']  # 南
    osbs_date = np.array(OSBS['date'].to_numpy())
    osbs_count = np.array(OSBS['total_count'].to_numpy())

    sum_monthly = np.zeros((12, 2))

    for i in range(len(tool_date)):
        mm = int(tool_date[i].split('/')[1])
        sum_monthly[mm - 1, 0] = sum_monthly[mm - 1, 0] + tool_count[i]
    for i in range(len(osbs_date)):
        mm = int(osbs_date[i].split('/')[1])
        sum_monthly[mm - 1, 1] = sum_monthly[mm - 1, 1] + osbs_count[i]
    # 设置颜色
    colors = ['blue', 'green']

    # 设置每个横坐标位置的宽度
    bar_width = 0.35

    # 遍历每个月份，为每个月份添加两个数据的柱子
    for i in range(sum_monthly.shape[0]):
        x_values = np.arange(2) + i * (2 * bar_width)
        plt.bar(x_values, sum_monthly[i, :], width=bar_width, label=f'{i + 1}', color=colors)
    plt.xticks(np.arange(sum_monthly.shape[0]) * (2 * bar_width) + bar_width * 2.4,
               [f'{i + 1}' for i in range(sum_monthly.shape[0])])

    # 设置横坐标标签位置和标签
    plt.xlabel('month')
    plt.ylabel('population')

    # 添加图例，并指定图例内容
    legend_labels = ['TOOL', 'OSBS']
    legend_colors = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]

    # 添加图例，并指定图例内容和颜色
    plt.legend(handles=legend_colors, loc='upper right')

    # 移除右上角的图例
    # plt.legend().set_visible(False)
    # 显示柱状图
    plt.savefig("count/f_a_count")


def f_a_rate(df):
    TOOL = df[df['location'] == 'TOOL']  # 北
    tool_date = np.array(TOOL['date'].to_numpy())
    tool_rate = np.array(TOOL['total_rate'].to_numpy())
    OSBS = df[df['location'] == 'OSBS']  # 南
    osbs_date = np.array(OSBS['date'].to_numpy())
    osbs_rate = np.array(OSBS['total_rate'].to_numpy())

    sum_monthly = np.zeros((12, 2))

    for i in range(len(tool_date)):
        mm = int(tool_date[i].split('/')[1])
        temp = sum_monthly[mm - 1, 0]
        if tool_rate[i] > temp:
            sum_monthly[mm - 1, 0] = tool_rate[i]

    for i in range(len(osbs_date)):
        mm = int(osbs_date[i].split('/')[1])
        temp = sum_monthly[mm - 1, 1]
        if osbs_rate[i] > temp:
            sum_monthly[mm - 1, 1] = osbs_rate[i]
    # 设置颜色
    colors = ['blue', 'green']

    # 设置每个横坐标位置的宽度
    bar_width = 0.35

    # 遍历每个月份，为每个月份添加两个数据的柱子
    for i in range(sum_monthly.shape[0]):
        x_values = np.arange(2) + i * (2 * bar_width)
        plt.bar(x_values, sum_monthly[i, :], width=bar_width, label=f'{i + 1}', color=colors)
    plt.xticks(np.arange(sum_monthly.shape[0]) * (2 * bar_width) + bar_width * 2.4,
               [f'{i + 1}' for i in range(sum_monthly.shape[0])])

    # 设置横坐标标签位置和标签
    plt.xlabel('month')
    plt.ylabel('positive rate')

    # 添加图例
    # 添加图例，并指定图例内容
    legend_labels = ['TOOL', 'OSBS']
    legend_colors = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]

    # 添加图例，并指定图例内容和颜色
    plt.legend(handles=legend_colors, loc='upper right')

    def percentage_formatter(x, pos):
        return f'{x:.2%}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    # 移除右上角的图例
    # plt.legend().set_visible(False)
    # 显示柱状图
    plt.savefig("rate/f_a_rate")


def gen_linear_regression_count_rate():
    # Reads a CSV file named "count_rate.csv" into a DataFrame 'df'
    df = pd.read_csv("count_rate.csv")
    # Filters rows where the 'total_rate' column is not equal to 0 and extracts the required data using the 'np_list' function. This data is stored in variables with suffix '_n0'.
    selected_data = df[df['total_rate'] != 0]
    wind_speed_n0, humidity_n0, pressure_n0, precipitation_n0, temperature_n0, rate_n0, count_n0 = np_list(
        selected_data)
    # Calculates the length of the selected data and multiplies it by 2 to use it for slicing another subset from the DataFrame where 'total_rate' is equal to 0. This subset is then processed similarly, and the data is stored in variables with suffix '_e0'.
    l = len(selected_data) * 2
    selected_data = df[df['total_rate'] == 0]
    wind_speed_e0, humidity_e0, pressure_e0, precipitation_e0, temperature_e0, rate_e0, count_e0 = np_list(
        selected_data)

    # Concatenates the data from both subsets ('_n0' and '_e0') to create new arrays ('*_new' variables).
    temperature_new = np.concatenate((temperature_n0, temperature_e0[:l]))
    humidity_new = np.concatenate((humidity_n0, humidity_e0[:l]))
    pressure_new = np.concatenate((pressure_n0, pressure_e0[:l]))
    precipitation_new = np.concatenate((precipitation_n0, precipitation_e0[:l]))
    wind_speed_new = np.concatenate((wind_speed_n0, wind_speed_e0[:l]))
    rate_new = np.concatenate((rate_n0, rate_e0[:l]))
    count_new = np.concatenate((count_n0, count_e0[:l]))

    # This function, 'linear_regression_summary', is designed to perform linear regression analysis.
    # linear_regression_summary(temperature_new, humidity_new, pressure_new, precipitation_new, wind_speed_new, count_new)
    # print("*********************************************************")
    # linear_regression_summary(temperature_new, humidity_new, pressure_new, precipitation_new, wind_speed_new, rate_new)

    # These functions visualize the relationship between weather and the positivity rate of a disease using a linear regression model.
    # They plot the actual data points, the fitted regression line, and provide a regression equation for interpretation
    # rate_func_no_sdr(wind_speed_new, rate_new)
    # count_func_pressure_no_sdr(pressure_new, count_new)
    # count_func_humidity_no_sdr(humidity_new, count_new)
    # count_func_windspeed_no_sdr(wind_speed_new, count_new)
    # season_rate(df)
    # season_count(df)
    latitude_count(df)
    # latitude_rate(df)
    # f_a_count(df)
    # f_a_rate(df)


def main():
    gen_linear_regression_count_rate()


if __name__ == "__main__":
    main()
