import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.covariance.robust_covariance import RobustCovariance

# 从csv文件中读取数据
data = pd.read_csv('data.csv')

# 定义SEM模型
model = """
Cell_Count ~ Specific_Conductance + Water_Temp + Dissolved_O2
Specific_Conductance ~ Dissolved_O2 + Water_Temp  
"""

# 拟合SEM模型
sem_model = sm.ols(model, data=data).fit()

# 显示模型概要
print(sem_model.summary())

# 显示标准化系数
print(sem_model.params)

# 计算方差膨胀因子,检查多重共线性
vif = pd.DataFrame()
vif["Features"] = sem_model.model.exog_names
vif["VIF"] = [variance_inflation_factor(sem_model.model.exog, i) for i in range(sem_model.model.exog.shape[1])]
print(vif)

# 模型假设检验
print(sem_model.f_test()) # F检验
print(sem_model.t_test()) # t检验

# 计算健壮标准误
rc = RobustCovariance(sem_model.model)
robust_cov = rc.cov
print(robust_cov)

# 添加路径改进模型
# 方法:重复进行回归,添加不同的自变量组合