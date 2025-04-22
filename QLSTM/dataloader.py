# 将数据进一步处理
from data import X, Y   # 导入处理好的X与Y
import numpy as np
from sklearn.model_selection import train_test_split

import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def sliding_window(X: np.ndarray, Y: np.ndarray, step: int = 30):
    if len(X) != len(Y):
        raise IndexError
    x, y = [], []
    for i in range(len(X) - step):
        # print(X[i:i + step, ])
        x.append(X[i:i + step, ])
        y.append(Y[i + step, 0])

    return np.array(x), np.array(y)

def flatten(X: np.ndarray):
    """将第二个维度和第三个维度进行压缩，用于随机森林

    Args:
        X (np.ndarray): 模型输入

    Returns:
        np.ndarray: 展平后的数组
    """
    return X.reshape(X.shape[0], -1)


# 使用滑动窗口，获得新的X，Y
X, Y = sliding_window(X, Y, 30)

# 两次拆分，获得 训练集、测试集、验证集 7:2:1 
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=1/3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
