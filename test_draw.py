import matplotlib.pyplot as plt

# 数据
days = [1, 2, 3, 4, 5, 6, 7]
mosquito_count = [100, 120, 90, 110, 80, 130, 70]
temperature = [25, 26, 24, 23, 27, 25, 28]
humidity = [60, 55, 70, 75, 50, 65, 45]
pressure = [1010, 1005, 1012, 1008, 1015, 1010, 1018]
precipitation = [0, 0, 1, 2, 0.5, 0, 3]
radiation = [200, 180, 220, 210, 230, 200, 240]

# 设置画布大小
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制左侧纵坐标轴（蚊子数量）
ax1.plot(days, mosquito_count, 'b-', label='Mosquito Count')
ax1.set_xlabel('Time')
ax1.set_ylabel('Mosquito Count', color='b')
ax1.tick_params('y', colors='b')

# 创建右侧纵坐标轴
ax2 = ax1.twinx()

# 绘制右侧纵坐标轴（温度）
ax2.plot(days, temperature, 'r-', label='Temperature')
ax2.set_ylabel('Temperature (°C)', color='r')
ax2.tick_params('y', colors='r')

# 创建第二个右侧纵坐标轴
ax3 = ax1.twinx()

# 位置调整，使得第二个右侧纵坐标轴共享横坐标轴
ax3.spines['right'].set_position(('outward', 35))

# 绘制右侧纵坐标轴（湿度）
ax3.plot(days, humidity, 'g-', label='Humidity')
ax3.set_ylabel('Humidity (%)', color='g')
ax3.tick_params('y', colors='g')

# 创建第三个右侧纵坐标轴
ax4 = ax1.twinx()

# 位置调整，使得第三个右侧纵坐标轴共享横坐标轴
ax4.spines['right'].set_position(('outward', 75))

# 绘制右侧纵坐标轴（气压）
ax4.plot(days, pressure, 'm-', label='Pressure')
ax4.set_ylabel('Pressure (hPa)', color='m')
ax4.tick_params('y', colors='m')

# 创建第四个右侧纵坐标轴
ax5 = ax1.twinx()

# 位置调整，使得第四个右侧纵坐标轴共享横坐标轴
ax5.spines['right'].set_position(('outward', 120))

# 绘制右侧纵坐标轴（降水）
ax5.plot(days, precipitation, 'c-', label='Precipitation')
ax5.set_ylabel('Precipitation (mm)', color='c')
ax5.tick_params('y', colors='c')

# 创建第五个右侧纵坐标轴
ax6 = ax1.twinx()

# 位置调整，使得第五个右侧纵坐标轴共享横坐标轴
ax6.spines['right'].set_position(('outward', 160))

# 绘制右侧纵坐标轴（辐射）
ax6.plot(days, radiation, 'y-', label='Radiation')
ax6.set_ylabel('Radiation', color='y')
ax6.tick_params('y', colors='y')

# 显示图例
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# 显示图形
plt.show()
