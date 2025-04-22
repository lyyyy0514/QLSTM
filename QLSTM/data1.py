# 加载数据到内存当中，并做第一步处理
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.read_excel(r"D:\Desktop\学习\shzhishu.xlsx", engine="openpyxl")

X = data[['chgpct', 'trade','trdsum']]
Y = data[['close']]

y_log = np.log(Y / Y.shift(1))
Y = y_log.dropna()

# 获取波动率
model = arch_model(Y, vol='Garch', p=1, q=1)
results = model.fit()
volatility = results.conditional_volatility

# 丢弃 X 的第一行，使其长度与 volatility 一致，并将Volatility加入到X当中作为新的一列
X = X.iloc[1:]
X['Volatility'] = volatility.values

# 分别对X，Y进行归一化操作
scaler_x = MinMaxScaler(feature_range=(0,1)) 
scaler_y = MinMaxScaler(feature_range=(0,1))  
X = scaler_x.fit_transform(X)   # shape 1759 * 5
Y = scaler_y.fit_transform(Y)   # shape 1759 * 1
