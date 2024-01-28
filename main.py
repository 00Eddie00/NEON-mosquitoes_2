import pandas as pd

# 读入CSV数据
df = pd.read_csv('NEON.D02.SERC.DP1.10041.001.mos_pathogenresults.2015-06.basic.20221122T183932Z.csv')
# 数据预处理
df['testResult'] = df['testResult'].apply(lambda x: x.strip())

# 按样本ID聚合数据
grouped = df.groupby('testingVialID')


# 定义判断样本阳性的函数
def is_positive(group):
    for i, row in group.iterrows():
        if row['testResult'] == 'Positive':
            return True
    return False


# 定义判断样本中病原体阳性的函数
def get_positives(group):
    positives = []
    for i, row in group.iterrows():
        if row['testResult'] == 'Positive':
            positives.append(row['testPathogenName'])
    return positives


# 聚合判定每个样本的结果
results = {}
for name, group in grouped:
    results[name] = 'Positive' if is_positive(group) else 'Negative'

# 统计各病原体阳性数和阳性率
pathogen_positives = {}
for name, group in grouped:
    for pathogen in get_positives(group):
        if pathogen not in pathogen_positives:
            pathogen_positives[pathogen] = 0
        pathogen_positives[pathogen] += 1
total = len(results)
for pathogen, count in pathogen_positives.items():
    rate = count / total
    print(pathogen, "positive rate:", rate)

# 统计并输出总结果
positive = sum(v == 'Positive' for v in results.values())
print("样本个数:", total)
print("其中阳性个数:", positive)
print("阳性率:", positive / total)

